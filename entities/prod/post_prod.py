from sqlalchemy import Column, Integer, String, ForeignKey, Date, UUID
from sqlalchemy.orm import relationship
from core.database import BaseProd

class PostProd(BaseProd):
    __tablename__ = "post"

    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(Date, nullable=False)
    text = Column(String, nullable=False)
    author_id = Column(Integer, ForeignKey("user_entity.id"), nullable=False)

    author = relationship("UserProd", back_populates="posts")
