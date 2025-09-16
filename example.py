"""
Document Verification Expert Agent

A specialized agent that demonstrates advanced SDK features:
- Scope detection: Determines if questions are within its expertise
- Structured responses: Returns JSON-formatted answers for document queries
- Transfer capability: Uses TransferToBlockResponse for out-of-scope questions
- Internal system simulation: Connects to a non-public expert system for document tracking
- LLM-powered analysis: Uses structured content generation for intent detection

This agent represents a realistic use case where an internal expert system
(that cannot expose a public API) provides document verification requirements
through an intelligent conversational interface.
"""

import os
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    APIKeyAuth,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    LLMConfig,
    OpenAIProviderConfig,
    TransferToBlockResponse,
)
from zowie_agent_sdk.domain import BearerTokenAuth


class DocumentType(str, Enum):
    """Types of documents required for verification"""

    ID_CARD = "id_card"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    PROOF_OF_RESIDENCE = "proof_of_residence"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    EMPLOYMENT_LETTER = "employment_letter"
    TAX_RETURN = "tax_return"


class DocumentStatus(str, Enum):
    """Status of individual documents"""

    NOT_SUBMITTED = "not_submitted"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Document(BaseModel):
    """Individual document information"""

    type: DocumentType
    status: DocumentStatus
    submitted_date: Optional[str] = None
    review_notes: Optional[str] = None
    expiry_date: Optional[str] = None


class QueryAnalysis(BaseModel):
    """Analysis of user query to determine scope and intent"""

    is_document_related: bool = Field(
        description="Whether the query is about document verification, requirements, or status"
    )
    specific_topic: Optional[str] = Field(
        description="Specific document topic if identified (e.g., 'passport status')"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence that this query is within agent's scope"
    )
    requires_status_check: bool = Field(
        description="Whether we need to check the user's current document status"
    )
    user_question: str = Field(description="Summary of what the user is asking")


class DocumentRequirements(BaseModel):
    """Complete document requirements from internal expert system"""

    user_id: str
    account_type: str
    required_documents: List[Document]
    verification_deadline: Optional[str]
    overall_status: Literal["incomplete", "pending_review", "approved", "rejected"]


class DocumentAnswer(BaseModel):
    """Direct response format for document-related queries"""

    message: str = Field(description="Direct response message to user")
    missing_documents: Optional[List[str]] = Field(
        default=None, description="List of documents still needed"
    )
    rejected_documents: Optional[List[str]] = Field(
        default=None, description="List of rejected documents that need resubmission"
    )
    next_steps: Optional[List[str]] = Field(
        default=None, description="Clear next steps for the user"
    )


class DocumentVerificationExpertAgent(Agent):
    """Expert agent for document verification requirements."""

    def handle(self, context: Context) -> AgentResponse:
        """Process requests related to document verification."""

        if not context.messages:
            return ContinueConversationResponse(
                message="Hello! I'm the Document Verification Expert. I can help you understand:\n"
                "• What documents you need to submit\n"
                "• The status of documents you've already submitted\n"
                "• Why a document might have been rejected\n"
                "• Next steps for completing your verification\n\n"
                "What would you like to know about your document requirements?"
            )

        # Step 1: Analyze if the query is within scope (document-related)
        query_analysis = context.llm.generate_structured_content(
            messages=context.messages,
            schema=QueryAnalysis,
            system_instruction=(
                "Analyze if this query is about document verification, document requirements, "
                "document status, or the verification process. Be strict: only mark as "
                "document_related if it's specifically about documents needed for account "
                "verification. General account questions, technical support, orders, passwords, "
                "business hours, or other non-document topics should be marked as NOT "
                "document_related."
            ),
        )

        # Log analysis for debugging
        self.logger.info(
            f"Query analysis - Topic: {query_analysis.specific_topic}, "
            f"Document related: {query_analysis.is_document_related}, "
            f"Confidence: {query_analysis.confidence}"
        )

        # Step 2: If out of scope, transfer to general support
        if not query_analysis.is_document_related or query_analysis.confidence < 0.7:
            self.logger.info(
                f"Query out of scope. Topic: {query_analysis.specific_topic}, "
                f"Confidence: {query_analysis.confidence}"
            )

            # Generate a polite transfer message
            transfer_message = (
                f"I specialize in document verification requirements, but your question "
                f"about '{query_analysis.user_question}' is outside my area of expertise. "
                f"Let me transfer you to our general support team who can better assist you."
            )

            return TransferToBlockResponse(
                message=transfer_message,
                next_block="general-support",  # Reference key for general support block
            )

        # Step 3: Query is in scope - get document requirements and generate response
        try:
            document_requirements = self._get_document_requirements(context)

            # Generate natural conversational response
            response = context.llm.generate_content(
                messages=context.messages,
                system_instruction=self._build_system_instruction(
                    query_analysis, document_requirements
                ),
            )

            return ContinueConversationResponse(message=response.text)

        except Exception as e:
            self.logger.error(f"Failed to process document query: {e}")
            return ContinueConversationResponse(
                message="I'm having trouble accessing the document verification system. "
                "Please try again in a moment, or contact support if the issue persists."
            )

    def _get_document_requirements(self, context: Context) -> DocumentRequirements:
        """
        Simulate querying internal expert system for document requirements.

        MOCK API DEMONSTRATION:
        This method demonstrates connecting to an internal expert system that:
        1. Cannot expose a public API for security/business reasons
        2. Contains complex business logic only developers understand
        3. Provides document verification requirements for users

        In production, this would be a real HTTP call to an internal service like:
        - POST https://internal-verification-system.company.com/api/v1/users/{user_id}/requirements
        - With proper authentication, error handling, and business logic

        For demo purposes, we use a public random number API to simulate different
        document scenarios and business rules.
        """
        user_id = context.metadata.conversationId

        # MOCK: Use external random number service to simulate internal API response
        # This represents calling your internal expert system that knows about:
        # - User's account type and verification level required
        # - Which documents have been submitted and their review status
        # - Business rules for what documents are needed for each account tier
        response = context.http.get(
            url="https://csrng.net/csrng/csrng.php?min=1&max=100",
            headers={"User-Agent": "DocumentExpertSystem/1.0"},
            timeout_seconds=5,
        )

        # MOCK: Extract random seed to simulate different business scenarios
        # In reality, this would be parsing complex document status from your system
        seed = response.json()[0]["random"] if response.status_code == 200 else 50

        # MOCK: Generate realistic document requirements based on seed
        # In production, this data would come from your internal database/system
        documents = []

        # MOCK BUSINESS LOGIC: Always require government-issued ID
        # Seed > 30 means user already submitted and it was approved
        documents.append(
            Document(
                type=DocumentType.ID_CARD,
                status=DocumentStatus.APPROVED if seed > 30 else DocumentStatus.NOT_SUBMITTED,
                submitted_date="2024-01-10T10:00:00Z" if seed > 30 else None,
            )
        )

        # MOCK BUSINESS LOGIC: Proof of residence requirements
        # Seed divisible by 3 = document was submitted but rejected (common scenario)
        if seed % 3 == 0:
            documents.append(
                Document(
                    type=DocumentType.PROOF_OF_RESIDENCE,
                    status=DocumentStatus.REJECTED,
                    submitted_date="2024-01-11T14:30:00Z",
                    review_notes="Address not clearly visible",
                )
            )
        else:
            # Otherwise, either not submitted (seed < 50) or approved (seed >= 50)
            documents.append(
                Document(
                    type=DocumentType.PROOF_OF_RESIDENCE,
                    status=DocumentStatus.NOT_SUBMITTED if seed < 50 else DocumentStatus.APPROVED,
                    submitted_date="2024-01-12T09:00:00Z" if seed >= 50 else None,
                )
            )

        # MOCK BUSINESS LOGIC: Premium accounts (seed > 60) require additional verification
        # This represents complex business rules only your internal system knows
        if seed > 60:
            documents.append(
                Document(
                    type=DocumentType.UTILITY_BILL,
                    status=DocumentStatus.NOT_SUBMITTED,
                    review_notes="Required for premium account verification",
                )
            )

        # MOCK BUSINESS LOGIC: Business accounts (seed % 5 == 0) need employment verification
        if seed % 5 == 0:
            documents.append(
                Document(
                    type=DocumentType.EMPLOYMENT_LETTER,
                    status=DocumentStatus.PENDING_REVIEW,
                    submitted_date="2024-01-13T16:00:00Z",
                )
            )

        # Determine overall status
        all_approved = all(d.status == DocumentStatus.APPROVED for d in documents)
        has_pending = any(d.status == DocumentStatus.PENDING_REVIEW for d in documents)
        has_rejected = any(d.status == DocumentStatus.REJECTED for d in documents)

        if all_approved:
            overall_status = "approved"
        elif has_rejected:
            overall_status = "rejected"
        elif has_pending:
            overall_status = "pending_review"
        else:
            overall_status = "incomplete"

        return DocumentRequirements(
            user_id=user_id,
            account_type="premium" if seed > 60 else "standard",
            required_documents=documents,
            verification_deadline=(datetime.now() + timedelta(days=30)).isoformat(),
            overall_status=overall_status,
        )

    def _build_system_instruction(
        self, query_analysis: QueryAnalysis, requirements: DocumentRequirements
    ) -> str:
        """Build system instruction for structured content generation."""

        # Identify missing and rejected documents
        missing = [
            doc.type.value
            for doc in requirements.required_documents
            if doc.status == DocumentStatus.NOT_SUBMITTED
        ]

        rejected = [
            f"{doc.type.value} ({doc.review_notes or 'needs resubmission'})"
            for doc in requirements.required_documents
            if doc.status == DocumentStatus.REJECTED
        ]

        return f"""
        You are a helpful document verification expert. The user is asking: {query_analysis.user_question}
        
        Current document status:
        - Overall status: {requirements.overall_status}
        - Account type: {requirements.account_type}
        - Missing documents: {missing or 'None'}
        - Rejected documents: {rejected or 'None'}
        - Deadline: {requirements.verification_deadline}
        
        Respond naturally and conversationally. Be helpful and specific about their document 
        requirements. If documents are missing or rejected, clearly explain what they need to do next.
        Keep the tone professional but friendly.
        """


def create_agent() -> DocumentVerificationExpertAgent:
    """Create and configure the document verification expert agent."""

    # Configure LLM provider
    llm_config: LLMConfig
    if os.getenv("GOOGLE_API_KEY"):
        llm_config = GoogleProviderConfig(
            api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.5-flash"
        )
        print("Using Google Gemini for document verification expert")
    elif os.getenv("OPENAI_API_KEY"):
        llm_config = OpenAIProviderConfig(api_key=os.environ["OPENAI_API_KEY"], model="gpt-5-mini")
        print("Using OpenAI GPT for document verification expert")
    else:
        raise ValueError("Please set GOOGLE_API_KEY or OPENAI_API_KEY environment variable")

    # Optional, though *strongly* recommended authentication
    auth_config = None
    if os.getenv("AGENT_API_KEY"):
        auth_config = BearerTokenAuth(
            token=os.environ["AGENT_API_KEY"],
        )
        print("API key authentication enabled")

    return DocumentVerificationExpertAgent(
        llm_config=llm_config, auth_config=auth_config, log_level="INFO"
    )


# Create the agent and expose globally
agent = create_agent()
app = agent.app
