import streamlit as st
import json
import time
import os
from io import BytesIO
from dotenv import load_dotenv
from pypdf import PdfReader

# Google GenAI imports
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the client with API key
try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        client = None
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")
    client = None

# Define the JSON schema required for CV data extraction (Step 1)
CV_EXTRACTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "candidate_name": types.Schema(type=types.Type.STRING, description="The full name of the candidate as listed on the CV."),
        "skills": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="A list of technical and soft skills strictly extracted from the CV."),
        "experience_years": types.Schema(type=types.Type.INTEGER, description="The total number of working experience years (rounded to the nearest integer). Use 0 if not mentioned."),
        "education": types.Schema(type=types.Type.STRING, description="The highest educational qualification mentioned, including the institution if possible."),
        "relevant_projects": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="Top two or three most relevant and detailed projects from the CV."),
    },
    required=["candidate_name", "skills", "experience_years", "education"],
)

# Define the JSON schema for scoring (Step 2)
SCORE_EXTRACTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "score": types.Schema(type=types.Type.INTEGER, description="A compatibility score between 0 and 100 based on the JD."),
        "justification": types.Schema(type=types.Type.STRING, description="A detailed justification explaining why this score was given, referencing specific skills or experience from the CV that match the JD."),
    },
    required=["score", "justification"],
)


# --- 1. LLM Chain Functions ---

def pdf_to_text(file_buffer):
    """Convert PDF file to raw text."""
    try:
        reader = PdfReader(file_buffer)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Step 1: Real Gemini API call for Structured Extraction
def extract_info_llm_call(cv_text, candidate_name):
    """
    Uses Gemini API (JSON Mode) to extract structured data from the CV text.
    """
    if not client:
        st.error("Gemini client not initialized. Check API key in .env.")
        return None

    st.info(f"LLM Step 1: Performing real structured data extraction for {candidate_name}...")
    
    # Enhanced Prompt for strict extraction
    prompt = (
        "You are a precise and objective data extraction machine. Analyze the following text and extract all "
        "required fields in JSON format. "
        "**Strict Rule:** You must NOT invent any information. If a field is not mentioned (like number of years), use the default value (0) or 'N/A' as specified in the schema. "
        "Do not use any external information or prior context. **Strictly adhere** to the text within the delimiters.\n\n"
        f"---BEGIN CV TEXT---\n{cv_text}\n---END CV TEXT---"
    )

    try:
        # Call Gemini specifying JSON Schema
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CV_EXTRACTION_SCHEMA,
                # System instruction to enforce strict adherence
                system_instruction="You are an expert data extraction machine. Output only valid JSON based *strictly* on the provided text input and schema. Ignore all prior context."
            )
        )
        
        # Parse the extracted JSON
        extracted_data = json.loads(response.text)
        return extracted_data

    except Exception as e:
        st.error(f"Gemini API call failed for data extraction for {candidate_name}. Error: {e}")
        # Return fallback data if real extraction fails
        return {
            "candidate_name": candidate_name,
            "skills": ["Extraction Failed - Check API Key or Input Format"],
            "experience_years": 0,
            "education": "N/A",
            "relevant_projects": ["N/A"]
        }


# Step 2: Real Gemini API call for Matching and Scoring 
def match_and_score_llm_call(jd_text, cv_json):
    """
    Uses Gemini API to compare the CV JSON data against the JD text and provide a score.
    """
    if not client:
        st.error("Gemini client not initialized. Check API key in .env.")
        return None

    candidate_name = cv_json.get('candidate_name', 'Unknown')
    st.info(f"LLM Step 2: Performing real scoring and matching for {candidate_name}...")

    # Combine the job description and the extracted CV data into a single prompt
    prompt = (
        "You are an expert HR analyst. Your task is to compare the Job Description (JD) against the Candidate's extracted CV data "
        "and provide a compatibility score (0-100) and a detailed justification in JSON format. "
        "Focus only on skills, experience, and education matching the JD. Ignore irrelevant information."
        "\n\n---JOB DESCRIPTION---\n"
        f"{jd_text}"
        "\n\n---CANDIDATE DATA (JSON)---\n"
        f"{json.dumps(cv_json, indent=2)}"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SCORE_EXTRACTION_SCHEMA,
                system_instruction="You are a professional HR scoring agent. Your output MUST be strictly valid JSON containing the score (integer 0-100) and justification for the compatibility between the provided JD and CV data."
            )
        )
        
        score_data = json.loads(response.text)
        
        # Structure the result to match the expected format of all_scores
        result = {
            "candidate_name": candidate_name,
            "score": score_data.get("score", 0),
            "justification": score_data.get("justification", "No justification provided by LLM.")
        }
        return result

    except Exception as e:
        st.error(f"Gemini API call failed for scoring {candidate_name}. Error: {e}")
        return {
            "candidate_name": candidate_name,
            "score": 0,
            "justification": "Scoring failed due to an API error."
        }


# Step 3: Mocks the LLM call for Ranking and Final Selection
def rank_candidates_llm_call(all_scores):
    """Mocks the final comparative analysis and ranking step."""
    st.info("LLM Step 3: Performing comparative analysis and final ranking...")
    time.sleep(3)
    
    # Simple sort for the mock data
    ranked = sorted(all_scores, key=lambda x: x['score'], reverse=True)
    best_candidate = ranked[0]
    
    mock_ranking_result = {
        "best_candidate": best_candidate['candidate_name'],
        "final_rationale": (
            f"Based on the analysis, **{best_candidate['candidate_name']}** was selected with a score of {best_candidate['score']}%. "
        ),
        "ranked_list": ranked
    }
    return mock_ranking_result

# --- 2. Streamlit UI and Workflow ---

st.set_page_config(page_title="LLM Candidate Matcher", layout="wide")

st.title("🤖 LLM Candidate Matcher")
st.markdown("Upload one Job Description (JD) and multiple Candidate CVs (PDF) to evaluate and rank them.")

# Display API key status
if client:
    st.sidebar.success("✅ Gemini Client Ready (API Key Loaded)")
else:
    st.sidebar.warning("⚠️ Gemini Client Not Ready. Please check GEMINI_API_KEY in .env.")

# File Uploaders
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Job Description")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd_upload")
    
with col2:
    st.header("2. Candidate CVs")
    cv_files = st.file_uploader(
        "Upload Multiple CVs (PDFs)", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="cv_upload"
    )

st.markdown("---")

if jd_file and cv_files and st.button("🚀 Start Evaluation & Ranking", type="primary"):
    
    if not client:
        st.error("Cannot start evaluation. Gemini API key is unavailable.")
        st.stop()
        
    st.header("📊 Evaluation Results")
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # 1. Process JD
    jd_text = pdf_to_text(jd_file)
    if not jd_text:
        st.error("Failed to extract text from Job Description.")
        st.stop()
        
    st.success("✅ Text extracted from Job Description.")
    progress_bar.progress(10, text="Job Description processed.")

    all_scores = []
    
    # 2. Process CVs (Looping the LLM Chain Steps 2 & 3)
    total_files = len(cv_files)
    
    for i, cv_file in enumerate(cv_files):
        candidate_name = cv_file.name.replace(".pdf", "").replace("_", " ").title()
        st.subheader(f"Analyzing Candidate: {candidate_name} ({i+1}/{total_files})")
        
        # A. PDF to Text
        cv_text = pdf_to_text(cv_file)
        if not cv_text:
            st.warning(f"Skipping {candidate_name} due to text extraction failure.")
            continue

        # B. LLM Step 1: Structured Extraction (Real)
        cv_json = extract_info_llm_call(cv_text, candidate_name)
        if cv_json is None:
            continue

        # C. LLM Step 2: Matching and Scoring (Real)
        score_result = match_and_score_llm_call(jd_text, cv_json)
        all_scores.append(score_result)
        
        # Display individual result
        st.metric(label=f"Match Score for {candidate_name}", value=f"{score_result['score']}%")
        with st.expander("Show Detailed Justification and Extracted Data"):
            st.write(score_result['justification'])
            st.markdown("#### Extracted Data (JSON):")
            st.json(cv_json)


        # Update progress bar
        progress = 10 + (80 * (i + 1) / total_files)
        progress_bar.progress(int(progress), text=f"Processing CVs: {i+1} of {total_files} completed.")

    # 3. LLM Chain Step 4: Final Ranking (Mock)
    if all_scores:
        st.markdown("---")
        st.header("🏆 Final Ranking and Selection")
        
        final_ranking = rank_candidates_llm_call(all_scores)
        
        # Display Final Selection
        st.success(f"Best Candidate Selected: **{final_ranking['best_candidate']}**")
        st.markdown(f"**Final Rationale:** {final_ranking['final_rationale']}")
        
        # Display Ranked List
        st.markdown("### Complete Ranked List")
        # ranked_df = [
        #     {"Rank": i+1, "Candidate": item['candidate_name'], "Score": item['score'], "Summary": item['justification'][:80] + "..."}
        #     for i, item in enumerate(final_ranking['ranked_list'])
        # ]

        ranked_df = [
            {"Rank": i+1, "Candidate": item['candidate_name'], "Score": item['score'], "Summary": item['justification']}
            for i, item in enumerate(final_ranking['ranked_list'])
        ]
        st.dataframe(ranked_df, use_container_width=True)
        
        progress_bar.progress(100, text="Analysis Complete!")
    else:
        st.warning("No CVs were successfully processed for ranking.")