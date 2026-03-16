from sqlalchemy import Column, Integer, Date, UUID
from sqlalchemy.orm import relationship

from core.database import BaseProd


class UserProd(BaseProd):
    __tablename__ = "user_entity"

    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(Date, nullable=False)
    is_verified = Column(Integer, nullable=False)
    report_count = Column(Integer, nullable=False)

    posts = relationship("PostProd", back_populates="author")

    def __repr__(self) -> str:
        return f"User(id={self.id}, created_at={self.created_at}, is_verified={self.is_verified}, report_count={self.report_count}, posts={len(self.posts)})"
