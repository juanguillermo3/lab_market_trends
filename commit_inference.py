"""
Title: Commit Inference
Description: Commits the marginal effects and metadata of an inference exercise.  Structured around a Unit of Work (single-transaction) pattern.
Depends on: inferences_db.py
"""

from datetime import datetime
from inferences_db import Inference, MarginalEffect, init_db
from sqlalchemy.exc import SQLAlchemyError

def commit_inference(segment_keyword, start_date, end_date, sample_size, vocabulary_size, keywords):
    """Commits an inference exercise and its marginal effects to the database.

    Args:
        segment_keyword (str): The segment keyword for the inference.
        start_date (datetime): The start date of the inference.
        end_date (datetime): The end date of the inference.
        sample_size (int): The number of samples used.
        vocabulary_size (int): The vocabulary size used.
        keywords (list): A list of dictionaries with keys: 'keyword', 'marginal_effect', 'prop_doc'.

    Returns:
        str: A success or error message.
    """
    Session = init_db()
    session = Session()

    try:
        with session.begin():  # Start a transaction
            # Add a sample inference
            inference = Inference(
                segment_keyword=segment_keyword,
                start_date=start_date,
                end_date=end_date,
                sample_size=sample_size,
                vocabulary_size=vocabulary_size
            )
            session.add(inference)
            session.flush()  # Ensures inference.id is available for foreign keys

            # Add marginal effects for the inference
            for k in keywords:
                session.add(MarginalEffect(
                    inference_id=inference.id,
                    keyword=k["keyword"],
                    marginal_effect=k["marginal_effect"],
                    prop_doc=k["prop_doc"]
                ))

        return "Inference and marginal effects committed successfully!"

    except SQLAlchemyError as e:
        session.rollback()  # Rollback if anything goes wrong
        return f"Transaction rolled back due to error: {e}"

    finally:
        session.close()
