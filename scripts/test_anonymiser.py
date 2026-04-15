
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "scripts"))

from anonymiser import anonymiser_mixte_final, identifier_pii

test_cases = [
    ("Bonjour, je m'appelle Paul.", "Simple name"),
    ("\nPaul a une douleur à la poitrine.", "Newline before name"),
    ("Mon nom est Monsieur Paul Martin.", "Full name with title"),
    ("Contactez-moi au 06 12 34 56 78.", "French phone number"),
    ("Je vis au 123 Main Street, New York.", "US address"),
    ("Mon email est test.user@gmail.com.", "Email address"),
    ("Hospital de la Pitié Salpêtrière.", "Hospital name"),
    ("Le syndrome de Freeman Sheldon est rare.", "Medical syndrome (should be protected)"),
    ("Freeman Sheldon a été admis ce matin.", "Syndrome name at start of sentence"),
    ("Il prend du Paracétamol et de l'Amoxicilline.", "Drugs (should be allowed)"),
    ("Le patient Paul Freeman a été vu.", "Name containing 'Freeman'"),
    ("Tu es un médecin urgentiste expert.", "Role words (should be protected)"),
    ("L'expert m'a conseillé de rester calme.", "Role words (should be protected)"),
]

def run_tests():
    print(f"{'INPUT':<45} | {'ANONYMIZED':<45} | {'CAUGHT WORDS'}")
    print("-" * 120)
    for text, desc in test_cases:
        anonymized = anonymiser_mixte_final(text)
        found = identifier_pii(text)
        caught_words = ", ".join([f"'{f['text']}' ({f['type']})" for f in found])
        
        display_input = text.replace("\n", "\\n")
        display_anon = anonymized.replace("\n", "\\n")
        
        print(f"{display_input:<45} | {display_anon:<45} | {caught_words}")

if __name__ == "__main__":
    run_tests()
