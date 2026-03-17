import os
import sys
from google import genai
from dotenv import load_dotenv

load_dotenv()

def test_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    print("Testing Gemini API Key: [KEY DETECTED]")
    
    try:
        client = genai.Client(api_key=api_key)
        # Using a very small prompt to minimize token usage
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Say 'API Key Working'",
        )
        print(f"RESPONSE: {response.text}")
        print("\n✅ SUCCESS: API Key is valid and working!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ FAILED: {error_msg}")
        
        if "401" in error_msg or "API_KEY_INVALID" in error_msg:
            print("\nSUGGESTION: Your API Key appears to be INVALID. Please check it on Google AI Studio.")
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("\nSUGGESTION: Your API Key is VALID, but you have exhausted your QUOTA (Rate Limit).")
            print("Wait a few minutes or check your usage at: https://aistudio.google.com/app/plan_usage")
        else:
            print("\nSUGGESTION: Check your internet connection or if the Gemini service is down.")

if __name__ == "__main__":
    test_key()
