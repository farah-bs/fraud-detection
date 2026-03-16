from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, Date
from sqlalchemy.orm import relationship

from core.database import BaseTrain


class PostTrain(BaseTrain):
    __tablename__ = "posts"

    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(BigInteger, nullable=False)
    created_at = Column(Date, nullable=False)
    content = Column(String, nullable=False)

    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    author = relationship("UserTrain", back_populates="posts", foreign_keys=[user_id])
