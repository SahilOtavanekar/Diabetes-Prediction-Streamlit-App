"""
Helper script to run Streamlit app
Use this if 'streamlit' command is not recognized
"""
import sys
import os

if __name__ == "__main__":
    try:
        # Method 1: Use streamlit's web CLI directly
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", "app.py"]
        stcli.main()
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        print("\nTrying alternative method...")
        try:
            # Method 2: Run as subprocess
            import subprocess
            result = subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], 
                                  check=False, capture_output=False)
            if result.returncode != 0:
                print(f"\nStreamlit exited with code {result.returncode}")
                print("\nPlease ensure Streamlit is installed:")
                print("python -m pip install streamlit")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("\nPlease try:")
            print("python -m pip install streamlit")
            print("Then run: python -m streamlit run app.py")

