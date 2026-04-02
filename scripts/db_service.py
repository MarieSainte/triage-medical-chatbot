import gspread
from oauth2client.service_account import ServiceAccountCredentials

class GSheetsDB:
    def __init__(self, credentials_file, spreadsheet_name):
        """Initialise la connexion à Google Sheets."""
        # Définir les autorisations
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Authentification
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        self.client = gspread.authorize(creds)
        
        # Ouvrir le fichier
        self.sheet = self.client.open(spreadsheet_name).sheet1
        
        # Initialiser les colonnes si la feuille est vide
        if not self.sheet.row_values(1):
            self.sheet.append_row(["date", 
                                   "id_cas",
                                   "prompt_final",
                                   "reponse",  
                                   "metadata_json",
                                   "symptomes_str"
                                   ])

    def add_interaction(self, id_cas, prompt, reponse, metadata_json, symptomes_str):
        """Ajoute une nouvelle ligne dans le Google Sheet."""
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [now, id_cas, prompt, reponse, metadata_json, symptomes_str]
        self.sheet.append_row(row)

    def get_all_interactions(self):
        """
        Récupère toutes les interactions et les transforme en dictionnaire 
        en suivant l'ordre de add_interaction.
        """
        import json
        print("Téléchargement des données depuis Google Sheets...")
        
        toutes_les_lignes = self.sheet.get_all_values()
        
        if len(toutes_les_lignes) <= 1:
            return []
            
        # On ignore l'en-tête (ligne 0)
        lignes_donnees = toutes_les_lignes[1:]
        liste_cas_cliniques = []
        
        for ligne in lignes_donnees:
            # Ordre basé sur row = [now, id_cas, prompt, reponse, metadata_json, symptomes_str]
            try:
                # On essaie de parser le JSON des métadonnées s'il existe
                metadata = {}
                if len(ligne) > 4 and ligne[4]:
                    metadata = json.loads(ligne[4])
            except:
                metadata = ligne[4] if len(ligne) > 4 else {}

            cas = {
                "id_cas": ligne[1] if len(ligne) > 1 else "",
                "prompt": ligne[2] if len(ligne) > 2 else "",
                "reponse_ideale": ligne[3] if len(ligne) > 3 else "",
                "metadata": metadata
            }
            liste_cas_cliniques.append(cas)
            
        print(f"✅ {len(liste_cas_cliniques)} cas récupérés avec succès !")
        return liste_cas_cliniques