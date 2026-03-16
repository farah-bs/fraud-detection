from sqlalchemy import Column, Integer, Date, Boolean, BigInteger
from sqlalchemy.orm import relationship

from core.database import BaseTrain


class UserTrain(BaseTrain):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(BigInteger, nullable=False)
    created_at = Column(Date, nullable=False)
    is_verified = Column(Integer, nullable=False)
    report_count = Column(Integer, nullable=False)
    number_of_followers = Column(Integer, nullable=False)
    number_of_followings = Column(Integer, nullable=False)
    is_fraud = Column(Boolean, nullable=False)

    posts = relationship("PostTrain", back_populates="author")

    def __repr__(self) -> str:
        return f"User(id={self.id}, uuid={self.uuid} created_at={self.created_at}, is_verified={self.is_verified}, report_count={self.report_count}, is_fraud={self.is_fraud} posts={len(self.posts)})"

