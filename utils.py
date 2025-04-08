from dateutil import parser
import subprocess



def extract_year(date_str: str):
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.year
    except ValueError:
        return None  # Return None if parsing fails

def is_ollama_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", "/home/pruderad/bin/ollama"])
        return True
    except subprocess.CalledProcessError:
        return False