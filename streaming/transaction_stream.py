"""
Real-Time Transaction Streaming Engine
=======================================
Provides async generator, sync generator, and mock Kafka-style queue for
continuous transaction simulation at a configurable rate.

Architecture
------------
  TransactionGenerator  ─── generates individual synthetic transactions
  MockKafkaQueue        ─── async producer/consumer queue (Kafka-like API)
  TransactionStream     ─── orchestrates rate-limited async/sync streaming

Usage — async streaming (e.g. in a background task):
    stream = TransactionStream(transactions_per_second=5, fraud_rate=0.01)
    async for txn in stream.stream_async(max_transactions=100):
        result = await api_client.post("/predict", data=txn)

Usage — sync streaming (Streamlit compatible):
    for batch in stream.stream_sync(max_transactions=50, batch_size=5):
        update_dashboard(batch)
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from collections import deque
from typing import AsyncGenerator, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class StreamedTransaction:
    """
    One synthetic credit card transaction ready for API submission.
    Field names match the TransactionInput schema exactly.
    """
    transaction_id:      str
    timestamp:           float
    time:                float    # seconds from start of window
    amount:              float
    V1:  float;  V2:  float;  V3:  float;  V4:  float
    V5:  float;  V6:  float;  V7:  float;  V8:  float
    V9:  float;  V10: float;  V11: float;  V12: float
    V13: float;  V14: float;  V15: float;  V16: float
    V17: float;  V18: float;  V19: float;  V20: float
    V21: float;  V22: float;  V23: float;  V24: float
    V25: float;  V26: float;  V27: float;  V28: float
    is_synthetic_fraud: bool = False

    def to_api_dict(self) -> dict:
        """Serialise to the format accepted by POST /predict."""
        return {
            "time":           self.time,
            "amount":         self.amount,
            "transaction_id": self.transaction_id,
            **{f"V{i}": getattr(self, f"V{i}") for i in range(1, 29)},
        }


# ── Generator ─────────────────────────────────────────────────────────────────

class TransactionGenerator:
    """
    Generates statistically realistic synthetic credit card transactions.

    Legitimate transactions are drawn from a general multivariate normal
    distribution.  Fraudulent transactions embed anomalous values in the
    PCA components that are empirically most discriminative (V1, V3, V4,
    V10, V12, V14, V17).
    """

    def __init__(self, fraud_rate: float = 0.003, seed: Optional[int] = None):
        """
        Args:
            fraud_rate: Probability that any given transaction is fraudulent.
            seed:       Optional RNG seed for reproducibility.
        """
        self.fraud_rate   = fraud_rate
        self._rng         = np.random.default_rng(seed)
        self._start_time  = time.time()

    def generate_one(self) -> StreamedTransaction:
        """Return one synthetic transaction."""
        is_fraud = self._rng.random() < self.fraud_rate
        elapsed  = time.time() - self._start_time

        V = self._rng.normal(0, 1.5, 28)

        if is_fraud:
            # Fraud-mode PCA anomalies
            V[0]  = self._rng.uniform(-10, -3)    # V1
            V[2]  = self._rng.uniform(-10, -1)    # V3
            V[3]  = self._rng.uniform(1,    8)    # V4
            V[9]  = self._rng.uniform(-6,  -1)    # V10
            V[11] = self._rng.uniform(-8,  -3)    # V12
            V[13] = self._rng.uniform(-12, -4)    # V14
            V[16] = self._rng.uniform(-8,  -2)    # V17
            # Bimodal fraud amounts: large cash-out OR micro card-testing
            if self._rng.random() < 0.6:
                amount = float(abs(self._rng.normal(350, 400)))
            else:
                amount = float(self._rng.uniform(0.01, 2.0))
            amount = min(amount, 25_000.0)
        else:
            amount = float(abs(self._rng.exponential(55.0)))
            amount = min(amount, 8_000.0)

        return StreamedTransaction(
            transaction_id=f"TXN-{uuid.uuid4().hex[:12].upper()}",
            timestamp=time.time(),
            time=elapsed,
            amount=round(amount, 2),
            **{f"V{i + 1}": round(float(V[i]), 6) for i in range(28)},
            is_synthetic_fraud=is_fraud,
        )

    def generate_batch(self, n: int) -> List[StreamedTransaction]:
        """Generate a batch of n transactions."""
        return [self.generate_one() for _ in range(n)]


# ── Mock Kafka queue ──────────────────────────────────────────────────────────

class MockKafkaQueue:
    """
    Async FIFO queue with Kafka-style producer/consumer API.

    In a production deployment this module can be replaced by a real
    confluent-kafka or aiokafka client without changing the consumer interface.
    """

    def __init__(self, topic: str = "transactions", max_size: int = 10_000):
        self.topic          = topic
        self._queue         = asyncio.Queue(maxsize=max_size)
        self._history: deque = deque(maxlen=2_000)
        self.total_produced = 0
        self.total_consumed = 0

    async def produce(self, message: dict) -> None:
        """Non-blocking produce; drops message if queue is full."""
        try:
            self._queue.put_nowait(message)
            self._history.append(message)
            self.total_produced += 1
        except asyncio.QueueFull:
            logger.warning("Kafka queue full — message dropped")

    async def consume(self) -> dict:
        """Block until a message is available."""
        msg = await self._queue.get()
        self.total_consumed += 1
        return msg

    def qsize(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return {
            "topic":          self.topic,
            "queue_size":     self.qsize(),
            "total_produced": self.total_produced,
            "total_consumed": self.total_consumed,
        }


# ── Stream orchestrator ───────────────────────────────────────────────────────

class TransactionStream:
    """
    Rate-limited transaction stream with async and sync interfaces.

    The async interface is suitable for FastAPI background tasks and asyncio
    event loops.  The sync interface is designed for Streamlit's blocking
    execution model.
    """

    def __init__(
        self,
        transactions_per_second: float = 2.0,
        fraud_rate:              float = 0.003,
        use_kafka_queue:         bool  = True,
        seed:                    Optional[int] = None,
    ):
        self.tps          = transactions_per_second
        self.generator    = TransactionGenerator(fraud_rate=fraud_rate, seed=seed)
        self.kafka_queue  = MockKafkaQueue() if use_kafka_queue else None
        self._running     = False

    @property
    def interval(self) -> float:
        """Time between transactions in seconds."""
        return 1.0 / max(self.tps, 0.01)

    def stop(self):
        self._running = False

    # ── Async interface ───────────────────────────────────────────────────────

    async def stream_async(
        self,
        max_transactions: Optional[int] = None,
    ) -> AsyncGenerator[StreamedTransaction, None]:
        """
        Async generator that yields transactions at the configured TPS rate.
        Yields until max_transactions is reached (or forever if None).
        """
        count = 0
        while max_transactions is None or count < max_transactions:
            txn = self.generator.generate_one()
            if self.kafka_queue:
                await self.kafka_queue.produce(txn.to_api_dict())
            yield txn
            count += 1
            await asyncio.sleep(self.interval)

    async def start_producer(self) -> None:
        """
        Background producer task — continuously pushes transactions to the
        Kafka queue.  Consume with `kafka_queue.consume()`.
        """
        self._running = True
        logger.info(f"Producer started at {self.tps} TPS")
        while self._running:
            txn = self.generator.generate_one()
            if self.kafka_queue:
                await self.kafka_queue.produce(txn.to_api_dict())
            await asyncio.sleep(self.interval)
        logger.info("Producer stopped")

    # ── Sync interface (Streamlit compatible) ─────────────────────────────────

    def stream_sync(
        self,
        max_transactions: Optional[int] = None,
        batch_size:       int = 1,
    ) -> Generator[List[StreamedTransaction], None, None]:
        """
        Synchronous generator that yields batches of transactions.
        Sleeps for `batch_size / tps` seconds between batches.

        Args:
            max_transactions: Stop after yielding this many total transactions.
            batch_size:       Number of transactions per yielded batch.
        """
        count    = 0
        sleep_t  = batch_size / max(self.tps, 0.01)

        while max_transactions is None or count < max_transactions:
            batch  = self.generator.generate_batch(batch_size)
            yield batch
            count += batch_size
            time.sleep(sleep_t)
