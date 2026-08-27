from fastapi import APIRouter, HTTPException, status
from schemas.nlp import (
    NLPAnalyzeRequest,
    NLPAnalyzeResponse,
    NLPBatchAnalyzeRequest,
    NLPBatchAnalyzeResponse,
)
from nlp.services.multilingual_pipeline import multilingual_pipeline

router = APIRouter(prefix="/nlp", tags=["Multilingual NLP"])


@router.post("/analyze", response_model=NLPAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_text(request: NLPAnalyzeRequest) -> NLPAnalyzeResponse:
    """
    Analyzes input text through the 7-stage unified Multilingual NLP pipeline.
    Executes script detection, language disambiguation, canonical normalization,
    and subword tokenization with zero external model dependencies.
    """
    try:
        result = multilingual_pipeline.analyze(request.text)
        return NLPAnalyzeResponse(**result.model_dump())
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP analysis error: {str(err)}",
        )


@router.post("/analyze/batch", response_model=NLPBatchAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_batch_texts(request: NLPBatchAnalyzeRequest) -> NLPBatchAnalyzeResponse:
    """
    Analyzes a batch of texts through the unified Multilingual NLP pipeline.
    """
    try:
        results = multilingual_pipeline.analyze_batch(request.texts)
        return NLPBatchAnalyzeResponse(
            results=[NLPAnalyzeResponse(**r.model_dump()) for r in results],
            total_texts=len(results),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch NLP analysis error: {str(err)}",
        )
