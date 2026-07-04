from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    agent_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    session = relationship("Session", back_populates="conversations")


class MedicalAnalysis(Base):
    __tablename__ = "medical_analyses"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    input_image_path = Column(String, nullable=True)
    result_summary = Column(Text, nullable=True)
    result_details = Column(JSON, nullable=True)
    validated = Column(Boolean, default=False)
    validation_result = Column(String, nullable=True)
    validation_comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class DatabaseManager:
    def __init__(self, db_path: str = "./data/medical_assistant.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def ensure_user(self, user_id: str):
        db = self.get_session()
        try:
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                user = User(id=user_id)
                db.add(user)
                db.commit()
            else:
                user.last_active = datetime.now()
                db.commit()
            return user
        finally:
            db.close()

    def ensure_session(self, session_id: str, user_id: str = None):
        db = self.get_session()
        try:
            session = db.query(Session).filter_by(id=session_id).first()
            if not session:
                session = Session(id=session_id, user_id=user_id)
                db.add(session)
                db.commit()
            else:
                session.last_active = datetime.now()
                db.commit()
            return session
        finally:
            db.close()

    def save_conversation(self, session_id: str, role: str, content: str, agent_name: str = None):
        db = self.get_session()
        try:
            conv_id = f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            conv = Conversation(
                id=conv_id,
                session_id=session_id,
                role=role,
                content=content,
                agent_name=agent_name,
            )
            db.add(conv)
            db.commit()
            return conv
        finally:
            db.close()

    def get_conversation_history(self, session_id: str, limit: int = 20):
        db = self.get_session()
        try:
            convs = (
                db.query(Conversation)
                .filter_by(session_id=session_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {"role": c.role, "content": c.content, "agent": c.agent_name, "time": c.created_at.isoformat()}
                for c in reversed(convs)
            ]
        finally:
            db.close()

    def save_medical_analysis(self, session_id: str, analysis_type: str, result_summary: str,
                              input_image_path: str = None, result_details: dict = None):
        db = self.get_session()
        try:
            analysis_id = f"ma_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            analysis = MedicalAnalysis(
                id=analysis_id,
                session_id=session_id,
                analysis_type=analysis_type,
                input_image_path=input_image_path,
                result_summary=result_summary,
                result_details=result_details,
            )
            db.add(analysis)
            db.commit()
            return analysis
        finally:
            db.close()

    def update_validation(self, analysis_id: str, validation_result: str, comments: str = None):
        db = self.get_session()
        try:
            analysis = db.query(MedicalAnalysis).filter_by(id=analysis_id).first()
            if analysis:
                analysis.validated = True
                analysis.validation_result = validation_result
                analysis.validation_comments = comments
                db.commit()
            return analysis
        finally:
            db.close()

    def get_session_analyses(self, session_id: str):
        db = self.get_session()
        try:
            analyses = (
                db.query(MedicalAnalysis)
                .filter_by(session_id=session_id)
                .order_by(MedicalAnalysis.created_at.desc())
                .all()
            )
            return [
                {
                    "id": a.id,
                    "type": a.analysis_type,
                    "summary": a.result_summary,
                    "validated": a.validated,
                    "validation_result": a.validation_result,
                    "time": a.created_at.isoformat(),
                }
                for a in analyses
            ]
        finally:
            db.close()