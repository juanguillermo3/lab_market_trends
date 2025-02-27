"""
Title: Inferences Database
Description: Provides a database model using SQLAlchemy to store marginal effects related to specific inference exercises.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from datetime import datetime

# Base class for model definitions
Base = declarative_base()

# Inferences Table
class Inference(Base):
    __tablename__ = 'inferences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_keyword = Column(String(100), nullable=False, index=True)  # New field
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    inference_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    sample_size = Column(Integer, nullable=False)
    vocabulary_size = Column(Integer, nullable=False)

    # Relationship with MarginalEffects
    marginal_effects = relationship(
        'MarginalEffect',
        back_populates='inference',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return (f"<Inference(id={self.id}, segment='{self.segment_keyword}', "
                f"start_date={self.start_date}, end_date={self.end_date}, "
                f"inference_date={self.inference_date}, sample_size={self.sample_size}, "
                f"vocabulary_size={self.vocabulary_size})>")


# Marginal Effects Table
class MarginalEffect(Base):
    __tablename__ = 'marginal_effects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    inference_id = Column(Integer, ForeignKey('inferences.id', ondelete='CASCADE'), nullable=False)
    keyword = Column(String(100), nullable=False, index=True)  # Indexed for search performance
    marginal_effect = Column(Float, nullable=False)
    prop_doc = Column(Float, nullable=False)

    # Relationship with Inference
    inference = relationship('Inference', back_populates='marginal_effects')

    def __repr__(self):
        return (f"<MarginalEffect(id={self.id}, keyword='{self.keyword}', "
                f"marginal_effect={self.marginal_effect}, prop_doc={self.prop_doc}, "
                f"inference_id={self.inference_id})>")


# Database Initialization Function
def init_db(database_url='sqlite:///inferences.db'):
    """Initialize SQLite database with SQLAlchemy."""
    engine = create_engine(database_url, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session
