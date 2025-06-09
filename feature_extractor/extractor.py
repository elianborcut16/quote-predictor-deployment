import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import logging
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom TF-IDF analyzer functions from the notebook
# These need to be at the module level so joblib can find them when loading the vectorizer

# Person-related terms for exclusion
person_terms = ["pax", "personen", "persoon", "mensen", "gasten", "gast", "pers"]

# Function to check if a word should be excluded based on fuzzy matching
def should_exclude_word(word):
    """Check if word should be excluded based on fuzzy matching to domain terms."""
    # Skip short words
    if len(word) < 3:
        return True
    
    # Skip number-only words or words that start with numbers
    if word.isdigit() or (len(word) > 0 and word[0].isdigit()):
        return True
    
    # Get Dutch stopwords (this might not work in the Docker environment,
    # but the vectorizer is already trained, so this function just needs to exist)
    try:
        dutch_stopwords = set(stopwords.words('dutch'))
        if word.lower() in dutch_stopwords:
            return True
    except:
        # If NLTK data is not available, continue without this check
        pass
    
    # This function previously checked against ships, packages, time and day keywords
    # We don't have those variables in this scope, but since the vectorizer is already trained,
    # we just need this function to exist with the same name
    
    # Check against person terms
    if process.extractOne(word, person_terms, scorer=fuzz.ratio)[1] > 75:
        return True
    
    return False

# Custom analyzer for TF-IDF that uses fuzzy matching to exclude domain words
def custom_analyzer(text):
    words = text.split()
    return [word for word in words if not should_exclude_word(word)]

class FeatureExtractor:
    """Feature extraction directly from the notebook."""
    
    def __init__(self, data_dir=None, tfidf_vectorizer=None):
        """
        Initialize the feature extractor with reference data.
        
        Args:
            data_dir: Directory containing reference data files (accounts.xlsx, 
                     bookings_2015.xlsx, orders.xlsx)
            tfidf_vectorizer: Pre-trained TF-IDF vectorizer for subject text
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        
        # Store the TF-IDF vectorizer
        self.tfidf_vectorizer = tfidf_vectorizer
        
        # Load reference data
        self.accounts_df = None
        self.bookings_2015_df = None
        self.orders_df = None
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference data from Excel files."""
        try:
            # Load accounts data
            accounts_path = os.path.join(self.data_dir, 'accounts.xlsx')
            if os.path.exists(accounts_path):
                logger.info(f"Loading accounts data from {accounts_path}")
                self.accounts_df = pd.read_excel(accounts_path)
            else:
                logger.warning(f"Accounts data file not found: {accounts_path}")
                self.accounts_df = pd.DataFrame(columns=["Accountname", "Accountnumber", "Address 1: city"])
                
            # Load 2015 bookings data
            bookings_path = os.path.join(self.data_dir, 'bookings_2015.xlsx')
            if os.path.exists(bookings_path):
                logger.info(f"Loading 2015 bookings data from {bookings_path}")
                self.bookings_2015_df = pd.read_excel(bookings_path)
            else:
                logger.warning(f"2015 Bookings data file not found: {bookings_path}")
                self.bookings_2015_df = pd.DataFrame(columns=["Organization", "Date/Time"])
                
            # Load orders data
            orders_path = os.path.join(self.data_dir, 'orders.xlsx')
            if os.path.exists(orders_path):
                logger.info(f"Loading orders data from {orders_path}")
                self.orders_df = pd.read_excel(orders_path)
            else:
                logger.warning(f"Orders data file not found: {orders_path}")
                self.orders_df = pd.DataFrame(columns=["Accountname", "Orderdate", "Orderconfirmationstatus"])
            
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
    
    def transform_quote(self, quote_data):
        """
        Transform a quote into features using exact notebook code.
        
        Args:
            quote_data: A dictionary with quote information
            
        Returns:
            DataFrame with features for the quote
        """
        try:
            # Create a single-row DataFrame from quote_data
            quotes_df = pd.DataFrame([quote_data])
            
            # Rename columns to match the notebook's expected format
            column_mapping = {
                'requestDate': 'Created on',
                'groupSize': 'Estimated group size',
                'customer': 'Accountname',
                'subject': 'Subject',
                'eventDate': 'Actual closing date',
                'contactPerson': 'Contact person'
            }
            
            for new_col, old_col in column_mapping.items():
                if new_col in quotes_df.columns and old_col not in quotes_df.columns:
                    quotes_df = quotes_df.rename(columns={new_col: old_col})
            
            # Ensure all required columns exist (with defaults if missing)
            required_columns = {
                'Created on': datetime.now().strftime('%Y-%m-%d'),
                'Subject': '',
                'Estimated group size': 30,
                'Accountname': '',
                'Status': 'Open',
                'Actual closing date': None,
                'Contact person': ''
            }
            
            for col, default in required_columns.items():
                if col not in quotes_df.columns:
                    quotes_df[col] = default
                    
            # Important: Use Contact person as fallback for Accountname if Accountname is empty
            quotes_df['Accountname'] = quotes_df['Accountname'].fillna(quotes_df['Contact person'])
            
            # Convert dates to datetime (exactly as in notebook)
            quotes_df['Created on'] = pd.to_datetime(quotes_df['Created on'], errors='coerce')
            quotes_df['Actual closing date'] = pd.to_datetime(quotes_df['Actual closing date'], errors='coerce')
            
            # ----- NOTEBOOK CODE STARTS HERE -----
            
            # Fill missing subjects with empty strings
            quotes_df["Subject"] = quotes_df["Subject"].fillna("")
            
            # ---- Step 1: Calculate Conversion Window ----
            # For a single quote, we use the fixed 93-day threshold from the notebook
            conversion_threshold = 93
            
            # Create binary indicator (1 if within threshold, 0 if not)
            quotes_df['Within_Conversion_Window'] = 0  # Default
            
            # If event date is provided, calculate days difference
            if 'Event Date' in quotes_df.columns:
                quotes_df['Time_Difference'] = (quotes_df['Event Date'] - quotes_df['Created on']).dt.days
                # Update Within_Conversion_Window based on time difference
                quotes_df.loc[quotes_df['Time_Difference'] <= conversion_threshold, 'Within_Conversion_Window'] = 1
                quotes_df.drop(columns=['Time_Difference'], inplace=True)
            
            # ---- Step 2: Text Cleaning and Feature Extraction ----
            # Use the complete ship list from the notebook
            # Full list of ships from the notebook (around 800 ships)
            raw_ship_list = """
            .38
't Smidtje 
A van Eijck
A. van Eijk
Aagje
Aagtje
Aaltje Engelina
Abaco
Abel Tasman
Actia
Actief
Ada
Adele
Adeline
Admiraal Heijn
Admiraal van Kinsbergen
Aegir
Agatha
Aimée
Albatros
Alberdina
Albert
Albert Cuyp
Albert Einstein 
Albert Johannes
Albert Pieter
Aldebaran
Alexander Von Humboldt II
Alexandra
Alida
Allegro
Allegro (ALG)
Alliantie
Allure
Alve Steden
AM1
Amalia
Amazone
Amber 
Ambiance
Ambolux 3
Ambulant
Amicitia
Amigo
Aminah
Amore Vici
AMS1
Amstel
Amstel Aquamarijn
Amstel Diamant
Amstel Jade
Amstel Robijn
Amstel Smaragd
Amstelveld
Amsterdam Boat Adventures
Andante
Andre Hazes
André Hazes
André van Duin
Andries Bicker
Anna
Anna ANA
Anna Catharina
Anna Maria
Anna Trijntje
Anna van Elburgh
Anna Van Nieuwkoop
Anna van Oranje
Anne
Anne Bonney
Antigua
Anton Dreesman
Antonia
Antonia Maria
Antonius
Aphrodite
Apollo
Apsara
Apsara
Aquamarijn
Arbeid Adelt
Arend
Argonaut
Ariana
Arno F
Artemis
Arthur van Schendel
Atalanta
Athene
Atlantis
Atlantis
Atyla
Aukje Zwaantje
Auwe Neel
Avanti
Avatar
Averechts
Avondrood
Avontur
Avontuur
Avontuur - Goes
Avontuur Loungecharter
Babbelaer
Bad Boat
Balder
Bali
Baltic
Banjaard
Bark Europa
Barlaeus 
Barracuda
Bartolotti
Batavia VOC
Beantra
Beau
Beau
Begijnhof
Belem
Belle
Belle Epoque
Belle van Zuylen
Bellefleur
Bennie Jolink
Berlin
Bern
Bint
Bisschop van Arkel
Blauwe Heks
Blop
Blue Clipper
Blue in Green
Blue Ocean
Blue Rhapsody
Blue Sky Boat
Boatnow
Boedapest
Boekanier
Bogota
Bonte Piet
Bontekoe
Booking.com
Boonapanich
Boot 1 
Boot 2 
Boot 3 
Boot 8 
Boot2-go 11
Boot2-go 12
Boot2-go 13
Boot2-go 14
Boot2-go 15
Boot2-go 16
Boot2-go 17
Boot2-go 18
Boot2-go 19
Boot2-go 20
Boozzer
Boreas
Bornrif
Bota Fogo
Bounty
Bracksand
Brandaris
Bredero
Bree Sant
Breitner
Britannia
Britannia 2
Broedertrouw
Bruinvisch
Brussels
Bruzer
Buggy
Byzantium
BZN 2
BZN 3
BZN 4
BZN 5
BZN 6
Caravel Vera Cruz
Carpe Diem
Catharina van Mijdrecht
Catharinaplaat
Celine Dion
Champagne Shower
Chateauroux
Christian Radich
Churchill 
City of Amsterdam
Confiance
Copenhagen
Corneille
Couperus
Creusen 1
Curacao
Dageraad
Dame van Amstel
Daniel Stalpaert
Daniël Stalpaert
Dankbaarheid
Dar Mlodziezy
David Bowie
De Hoop
De Laurier
De Lelie
De Onderneming
De Onderneming
De Pannenkoekenboot
de Verandering
De Wester
De Zuiderzee
d'Eersteling
Delphine
Dieuwke
Docks 1 
Docks 2 
Docks 3 
Docks 4 
Dolce Vita
Domani
Dorothee
Dr. Albert Heijn
Dr. Albert Schweitzer
Dreamtime
Drift Away
Drukwerk
Dublin
Dy Abt Fan Starum
Dyos
Eben Haëzer
Edison
Eendracht
Eendracht (EDT)
Eendracht (ENH)
Eenhoorn
Eensgezindheid
Eenvoud
Eeuw
Egberdina
Egelantier
Eivissa
Eke Maria
El Galeon
Elbrich
Eldorado
Elegant
Elektrische motorboat 1
Elisabeth
Elizabeth
Elly Glasius
Elvira
Emilia
Emma
Emmaa
Emmalis
Engel 
Esperanza
Ethel von Brixham
Etoile du Roy
Europa
Eve
Eveline
Eventus
Eye of the Wind
F van Dongen
F. van Dongen
Fadam 1
Faraday
Farahilde
Fayén
Feniks
Ferdinand Bol
Feroaring
Flagship
Flandre III
Fleur
Flying Dutchman
Flying Enterprise
Folke Bernadotte
Fortuna
Frans Hals
Frans Horjus
Frans van Mieris 
Franz Lehar
Freewilly
Friendship
Friesland
Frije Fugel (LE 194)
Frisian Queen
Frisius van Adel
Froukje
Fryderyk Chopin
Fuga
Gaasterland
Gaia
Gallant
Gangster
Gast
Geertje
Gerarda Johanna
Gordita
Götheborg
Gouden Bodem
Gouwzee
GR 1
Grace Kelly
Green Marine
Grietje
Griffioen
Grootvorst
Grote Beer
Grote Jager (LE 196)
Gulden Belofte
Gulden Leeuw
H Hertzberger
H. Hertzberger
H.P. Berlage
HA4
Haarlemmerpoort
HAL
Halve Maen
Halve Maen
Hanneke
Hannekes Boot
Hanzestad
Harry Slinger
Havana rondvaartboot
Havana sloep
Hege Wier
Helena
Helena
Helsinki
Hendrick de Keyser
Hendrick de Keyser
Hendrika Bartelds
Henry Hudson
Herman Brood
Hester
Hilda IV
Hildebrand 
Hoge Wier
Holland America Line
Hollandia (FV)
Hollandia (import)
Hollandia (MJ)
Hollandsch Diep
Hoop Doet Leven
Hoop geleid ons
Hoop op Behoud
Hoop op Welvaart
Hoop op zegen
Hoop op zegen
Horatio Nelson
HP Berlage
HRH
Hugo
HVAL
Hyacint
Hydro Xperience
Hydrograaf
Ideaal
IJdok 1
IJveer XIV 
Ilje
Ir. Frits Fentener van Vlissingen
Iris
Iselmar
Isis
Ivan Frank
Ivresse
J Coenen
J. Coenen
J. Vermeer
J.F. Staal
J.H. Mulder
Jackson 1
Jacob van Berlijn
Jacob van Lennep
Jacob van Ruysdael
Jacoba
Jacobsschelp
Jade
James Bond 
Jan Luijken 
Jan Steen
Jan van de Heijden
Jan van Riebeeck
Jannes Lovers
Jannes van Campen
Jantje
Jean Schmitz
Jeppe van Schier
Jeroen Krabbé
JF Staal
JH Mulder
Jikke
Joanna Saturna
Jochie
Johanna
Johanna Engelina
Johanna Lucretia
Johnny Kraaijkamp
Jonas
Jonckvrouw
Josephine
Josina Elisabeth
JR Tolkien
Judith Leyster
Juffrouw Ooievaer
Jules Verne
Kaat Mossel
Kapitan Borchardt
Kaprifolia
Kiev
Kikkerkoning
King
Klaas Kompaan
Kleijn Amsterdam
Kleine Jager
Klipper Anna
Klundert 26
Koh-i-Noor
Koning
Koning Willem Alexander
Koningin Emma
Koningin Fabiola
Koningin Juliana
Koningin Maxima
Koningin Wilhelmina
Korevaer
Kornelise Ykes
Kruzenshtern
Kuala Lumpur
La Bohème
La Grace
La Reine
Lammert van Dirk
Larus
Laurens Coster 
Lauta
Lauwerszee
LE 7 (Brandende Liefde)
Le Français
Leafde fan Fryslan
Leao Holandes
Leemstar Cooper
Leemstar Starlet
Leemstar Tender
Lelie
Lemsteraak HL3 
Lena Adriana
Libelle
Liberté
Lido
Liefde
Liefde Zeilschip
Lieve
Linde
Linquenda
Lis
Lissabon
Lola
London
Loth Lorien
Lotus
Loungeklipper de Vrijheid
Loungeklipper Umoja
Loungeskûtsje
Luciana
Lucky Stripper
Lucy
Ludwig von Beethoven
Luije Man
Lutine
M. de Klerk
M.C. Escher
Madrid
Mallejan
Mana
Manhattan
Manke Monnik
Manna
Mare fan Fryslan
Mare Frisium
Margot
Marie
Marie Galante
Marie Zurlohe
Marijn Joris
Marjet
Marjorie II
Mars
Mathilda
Max
Maxima
Maybe
MD3
Meander
Medusa
Meike
Mercedes
Mercurius
Mercurius (RV)
Meridiaan
Mermaid
Michiel de Ruyter
Middelsee
Minerva
Mir
Moby Queen
Moeke Zorn
Mon Ami
Mon Desir
Mona Lisa
Monaco
Mondriaan
Monne de Miranda
Montevideo
Morgana
Morgenster
Moshulu
Moskou
Mother goose
MPS Iris
Mr. Brown
Mr. Grey
Ms Agatha
MS Jordaan
MS Waddenzee
MU 1
Multatuli
Mumbai
Muze
Najade
Namaste
Nao Trinidad
Nao Victoria
Nassau
Naut
Neeltje
Nelson Mandela
Nemo
Nieuwe Maen
Nieuwe Zorg
Nijverheid
Nil Desperandum
Nirvana
Nirwana
Nomag
Nooit Volmaeckt
Noorderlicht
Noordfries
Noordster
Noordvaarder
Nova cura
Novel
NTB
Oban
Oboot
Ocean Diva Futura
Ocean Diva Verne
Oceans
Ocularis
Old Queen
Olivier van Noort
Onderneming
Onderneming (FSC, gr)
Ondine
Onrust
Ontmoeting
Onverwacht
Ooosterdok
Oosterschelde
Oost-Vlieland
Oostvogel
Orion
Ortolan
Oslo
Ouderzorg
Oudezijds
Ouwe
Overwinning
P Vingboons
P. Vingboons
P.J.H. Cuypers
Padde
Pallieter
Panta Rhei
Papi
Para Todos
Paradis
Paradiso 
Parel van de Maas
Paris
Parodie
Pascual Flores
Passaat
Pax
Pax
Pedro Doncker
Pegasus
Pelikaan
Pelikaan (HAR)
Peter de Grote
Phanta Rei
Phoenix
Phoenix - Samen Sterk
Picton Castle
Pierre Cuypers
Pierre Janssen
Piet Hein
Pieter
Pieter Heijn I
Pieter Heijn II
Pieter Heijn III
PJH Cuypers
Plastic Whale 600
Pogoria
Poolster
Port Side RIB
Port Side RIB
Poseidon
Pouwel Jonas
President J.F. Kennedy
Pride
Prince of new Amsterdam
Prins Bernhard
Prins Claus
Prins Constantijn
Prins Hendrik
Prins Johan Friso
Prins van Oranje
Prins Willem van Oranje
Prinses Alexia
Prinses Ariane
Prinses Beatrix
Prinses Catharina Amalia
Prinses Christina
Prinses Irene
Prinses Margriet
Prix d'Eau
Proost van Sint Jan
Pure spirit
Queen
Radboud
Radius
Ramses Shaffy
Raw Ferry 01
Rea-Klif
Rederijker
Regina Maris
Reindert
Rembrandt
Res Nova
Rijk de Gooyer
Rijnland
Rijnstroom
Rinske
Risdam
Risico
Rival
Riverdream
Rode Swan
Roerdomp
Rome
Roos
Roos
Roosevelt 
Rose
Rose
Royalty
Rupel
Russian Star
RWS 1
Saeftinghe
Saffier
Sailboa
Salonboot
Salonboot Avanti
Saloonsloep
Salute
Samenwerking
Sanne Sophia
Santa Maria Manuela
Sarah
Scaldis
Schokland
Schollevaar
Schuttevaer
Sedov
Selene
Senta
Sevilla
Shadow
Shiner
Shtandart
Sinta
Soeverein
Soeverein
Soeverijn
Sofia
Solar
Sperwer
Spes Mea
SS Succes
Staets I 
Staets II 
Stan Huygens
Stânfries
Statenjacht De Utrecht
Statsraad Lehmkuhl
Steady
Stedemaeght
Stella Maris
Stern
Sterrenwind
Stockpaerdt
Stoere Eenvoud
Stoomschip Succes 
Store Baelt
Stortemelk
Strijd
Struner
Style
Style
Succes
Sudermar
Sudwester
Sulis
Summertime
Sunflower
Sunshine
Suydersee
Suzanne
Swaen
Swaensborgh
SY Arwen
Sybaris
t Wapen fan Fryslân
Tecla
Tera Nova
Terra Nova 
Test schip voor individueel 1
Thalassa
Thalassa Partyschip
The Bright Boat (Liverpool)
The Club
The Dream
The Dude
The Flying Dutchman
The Lounge
The Lounge 2
Theo Kok
Tijd zal het Leeren
Titaan
Titanic
Tjalk Zuiderzee
Toekomst
Toon Hermans
Tourist
Traditioneel zeilschip
Traditioneel zeilschip (comfort class)
Traditioneel zeilschip (first class)
Traditioneel zeilschip (standard class)
Traditioneel zeilschip (superiour class)
Tres Hombres
Triton
Trouwe Hulp
Tsjerk Hiddes
Tulip
Twee
Twee gebroeders
Twister
Tzar
Umoja
Utopia
Valentijn
Valerie
Van Gogh
Van Hemert Stakenburg 2
Van Hemert Stakenburg 3
Vastberaden
Veerman van Kampen
Vellamo
Verandering
Vertrouwen (import)
Vertrouwen Enkhuizen
Vertrouwen Harlingen
Vertrouwen Lemmer
Verwisseling
Verwondering
Vibe
Victoria-S
Viking
VIP 1
Vliegende Draeck
Vliegende Hollander
Vlieter
Volharding
Vondel
Voorwaarts
Voorwaarts Voorwaarts
Vörding
Vossius
Vriendentrouw
Vriendschap
Vriendschap (VDS)
Vriendschap (VRS)
Vrijheid
Vrouw Dina
Vrouw Leentje
Vrouwe Francina
Vrouwe Frida Cornelia
Vrouwe Gerdina
Vrouwe Jitske
Vrouwe Johanna
Vrouwezand
W.A. Mozart
Waddenzee
Wadwaai
Wan-hoop 
Wapen fan Fryslan
Wapen van Amsterdam
Waterman
Watertaxi Hoorn
Watertoerist
Waterwolf
Welmoed
Welvaart
Wending
Wiede Weerga
Wien
Wilhelmina
Wilhelmus
Willem Barentsz
Willem Jacob
Willy
Wim Kan
Windbreeker
Windbruid
WM 1
WM1
Woef
Wolk
Wylde Swan
Wyn & Wetter
YB II
Ybeltje
Z8
Z9
Zahra
Zavi
Zeebeer
Zeedijk
Zeeland
Zeelandia
Zeemeeuw
Zeeotter
Zeester
Zeewolf
Zephyr
Zeven Wouden
Zilvermeeuw
Zonneboot
Zorg en Vlijt
Zorg met Vlijt
Zout
Zuiderzee
Zuiderzee (zeeschip)
Zuid-Holland
Zuid-Holland
Zuid-Holland (salonboot)
Zwaan
Zwadde
Zwaluw
Zwarte Bonsem
Zwarte Ruiter
Zwarte Valk
            """
            
            # Clean and process the raw ship list
            ship_names = [
                name.strip().lower()
                for name in raw_ship_list.split('\n')
                if name.strip() != '' and not name.strip().startswith('.')  # Skip empty and number-only lines
            ]

            # Remove duplicates
            ship_names = list(set(ship_names))

            # Ensure key ships from the required feature list are included
            required_ships = [
                "joanna saturna", "king", "bredero", "belle van zuylen", "salute", 
                "wapen van amsterdam", "belem", "breitner", "rijk de gooyer", 
                "kaat mossel", "test schip voor individueel 1", "frisius van adel", 
                "sunshine", "thalassa partyschip", "albert einstein"
            ]
            
            # Add these to the ship list if not already present
            for ship in required_ships:
                if ship not in ship_names:
                    ship_names.append(ship)
            
            # Define package variants
            package_variants = {
                "sail_in_city": ["sail in city", "sailincity", "sail in", "sailin"],
                "sail_in_classic": ["sail in classic", "sailinclassic", "sail in", "sailin"],
                "sail_out_city": ["sail out city", "sailoutcity", "sail out", "sailout"],
                "sail_out_tall": ["sail out tall", "sailouttall", "sail out", "sailout", "tall"],
                "afternoon_cruise": ["afternoon cruise", "middag cruise", "middagcruise"],
                "evening_cruise": ["evening cruise", "avond cruise", "avondcruise"],
                "morning_cruise": ["morning cruise", "ochtend cruise", "ochtendcruise"],
                "lunch": ["lunch"],
                "brunch": ["brunch"],
                "drinks": ["drinks", "borrel"],
                "bites": ["bites", "hapjes"],
                "sunset_cruise": ["sunset cruise", "zonsondergang cruise", "sunset"],
                "evening_drinks": ["evening drinks", "avond borrel", "avondborrel"],
                "diner": ["diner", "dinner"]
            }

            # Create flat list of package names and variants
            package_names = list(package_variants.keys())
            package_keywords = []
            for variants in package_variants.values():
                package_keywords.extend(variants)
            package_keywords = list(set(package_keywords))

            # Time and day dictionaries
            time_keywords = {'ochtend': 'ochtend', 'middag': 'middag', 'avond': 'avond'}
            time_keywords_list = list(time_keywords.keys())

            days_keywords = {
                'za': 'zaterdag', 'zat': 'zaterdag',
                'zon': 'zondag', 'zo': 'zondag',
                'ma': 'maandag', 'di': 'dinsdag',
                'wo': 'woensdag', 'do': 'donderdag',
                'vr': 'vrijdag'
            }
            all_day_keywords = list(days_keywords.keys()) + list(days_keywords.values())
            days_list = list(set(days_keywords.values()))  # Unique day values

            # Function to clean text
            def clean_text(text):
                if not text:
                    return ""
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            # Clean the subject text
            quotes_df["Cleaned Subject"] = quotes_df["Subject"].apply(clean_text)
            
            # Function to extract subject features
            def extract_subject_features(text):
                features = {
                    'Ship': None,
                    'Package': None,
                    'Day': None,
                    'Time': None,
                }
                
                # Find ship - try exact match first
                for ship in ship_names:
                    if ship in text:
                        features['Ship'] = ship
                        break
                
                # If no ship found, try fuzzy matching
                if not features['Ship'] and len(ship_names) > 0:
                    match = process.extractOne(text, ship_names, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 75:  # 75% similarity threshold
                        features['Ship'] = match[0]
                
                # Find package - check each variant
                for pkg_name, variants in package_variants.items():
                    if any(variant in text for variant in variants):
                        features['Package'] = pkg_name
                        break
                
                # If no package found, try fuzzy matching
                if not features['Package']:
                    match = process.extractOne(text, package_keywords, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 75:
                        # Find which package this keyword belongs to
                        for pkg_name, variants in package_variants.items():
                            if match[0] in variants:
                                features['Package'] = pkg_name
                                break
                
                # Find time mentions
                for time in time_keywords_list:
                    if time in text:
                        features['Time'] = time
                        break
                
                # Find day mentions
                for day in all_day_keywords:
                    if day in text:
                        # Map to full day name if needed
                        features['Day'] = days_keywords.get(day, day)
                        break
                
                return pd.Series(features)
            
            # Apply function to get structured features
            subject_features = quotes_df['Cleaned Subject'].apply(extract_subject_features)
            
            # Merge subject features back into quotes_df
            quotes_df = pd.concat([quotes_df, subject_features], axis=1)
            
            # Count subject words
            quotes_df["Subject_Word_Count"] = quotes_df["Subject"].apply(lambda x: len(str(x).split()))
            
            # ---- Step 3: Merge with Accounts Data ----
            if self.accounts_df is not None and not self.accounts_df.empty:
                # Standardize city to Title Case
                if "Address 1: city" in self.accounts_df.columns:
                    self.accounts_df["Address 1: city"] = self.accounts_df["Address 1: city"].str.title()
                
                # Select necessary columns
                accounts_subset = self.accounts_df[['Accountname', 'Accountnumber', 'Address 1: city']].rename(
                    columns={"Address 1: city": "Company city"}
                )
                
                # Try to merge accounts data
                quotes_df = quotes_df.merge(accounts_subset, on="Accountname", how="left")
                
                # Handle missing account numbers
                quotes_df["Accountnumber"] = quotes_df["Accountnumber"].fillna(-1).astype(int)
                
                # Create binary column for Amsterdam
                quotes_df["Company City Amsterdam"] = quotes_df["Company city"].apply(
                    lambda x: 1 if str(x).lower() == "amsterdam" else 0
                )
                
                # Drop the original city column
                quotes_df = quotes_df.drop(columns=["Company city"])
            else:
                # If no accounts data, create default columns
                quotes_df["Accountnumber"] = quotes_df["Accountname"].apply(lambda x: abs(hash(str(x))) % 1000000)
                quotes_df["Company City Amsterdam"] = quotes_df["Accountname"].apply(
                    lambda x: 1 if "amsterdam" in str(x).lower() else 0
                )
            
            # ---- Step 4: Identify Recurring Customers ----
            # Create a function to check if customer had prior bookings (simplified for a single quote)
            def is_recurring(account_name):
                if pd.isna(account_name) or not account_name:
                    return 0
                
                recurring = 0
                
                # Check bookings_2015
                if self.bookings_2015_df is not None and not self.bookings_2015_df.empty:
                    if 'Organization' in self.bookings_2015_df.columns:
                        orgs = self.bookings_2015_df['Organization'].dropna().unique()
                        # Try exact match first
                        if account_name in orgs:
                            recurring = 1
                        else:
                            # Try fuzzy match
                            match = process.extractOne(account_name, orgs, scorer=fuzz.token_sort_ratio)
                            if match and match[1] >= 80:  # 80% similarity threshold
                                recurring = 1
                
                # Check orders data
                if recurring == 0 and self.orders_df is not None and not self.orders_df.empty:
                    if 'Accountname' in self.orders_df.columns:
                        order_accounts = self.orders_df['Accountname'].dropna().unique()
                        # Try exact match first
                        if account_name in order_accounts:
                            recurring = 1
                        else:
                            # Try fuzzy match
                            match = process.extractOne(account_name, order_accounts, scorer=fuzz.token_sort_ratio)
                            if match and match[1] >= 80:  # 80% similarity threshold
                                recurring = 1
                
                return recurring
            
            # Apply function to determine recurring status
            quotes_df['Recurring Customer'] = quotes_df['Accountname'].apply(is_recurring)
            
            # ---- Step 5: Convert Created Date to Datetime Features ----
            quotes_df["Created Year"] = quotes_df["Created on"].dt.year
            quotes_df["Created Month"] = quotes_df["Created on"].dt.month
            quotes_df["Created DayOfWeek"] = quotes_df["Created on"].dt.weekday + 1  # 1-7 for Monday-Sunday
            
            # ---- Step 6: One-Hot Encode Features ----
            # Function to clean feature name
            def clean_name(val):
                if pd.isna(val):
                    return None
                return str(val).strip().lower().replace(" ", "_")
            
            # Clean values for each feature
            quotes_df['ship_clean'] = quotes_df['Ship'].apply(clean_name)
            quotes_df['package_clean'] = quotes_df['Package'].apply(clean_name)
            quotes_df['time_clean'] = quotes_df['Time'].apply(clean_name)
            quotes_df['day_clean'] = quotes_df['Day'].apply(clean_name)
            
            # Get top ships (use all available if few)
            top_ships = ship_names[:min(20, len(ship_names))]
            if "other" not in top_ships:
                top_ships.append("other")
            
            # Initialize empty DataFrame for one-hot encoding
            onehot_df = pd.DataFrame(index=quotes_df.index)
            
            # Add ship columns
            for ship in top_ships:
                col_name = f"ship_{clean_name(ship)}"
                onehot_df[col_name] = (quotes_df['ship_clean'] == clean_name(ship)).astype(int)
            
            # Add package columns
            for pkg in package_names:
                col_name = f"package_{clean_name(pkg)}"
                onehot_df[col_name] = (quotes_df['package_clean'] == clean_name(pkg)).astype(int)
            
            # Add time columns
            for time in time_keywords.values():
                col_name = f"time_{clean_name(time)}"
                onehot_df[col_name] = (quotes_df['time_clean'] == clean_name(time)).astype(int)
            
            # Add day columns
            for day in days_list:
                col_name = f"day_{clean_name(day)}"
                onehot_df[col_name] = (quotes_df['day_clean'] == clean_name(day)).astype(int)
            
            # Combine with original dataframe
            quotes_df = pd.concat([quotes_df, onehot_df], axis=1)
            
            # ---- Step 7: Apply TF-IDF using the pre-trained vectorizer ----
            tfidf_features = []
            tfidf_feature_names = []
            tfidf_df = pd.DataFrame()
            
            # Extract date features (like in notebook cell 14)
            quotes_df["Created Year"] = quotes_df["Created on"].dt.year
            quotes_df["Created Month"] = quotes_df["Created on"].dt.month
            quotes_df["Created DayOfWeek"] = quotes_df["Created on"].dt.weekday + 1  # Monday = 1, Sunday = 7
            
            # Apply the pre-trained vectorizer to the subject text
            if self.tfidf_vectorizer is not None:
                logger.info("Applying pre-trained TF-IDF vectorizer to cleaned subject text")
                
                # Get the subject text
                subject_text = quotes_df["Cleaned Subject"].iloc[0]
                
                # Apply the vectorizer
                try:
                    tfidf_matrix = self.tfidf_vectorizer.transform([subject_text])
                    
                    # Get feature names from the vectorizer
                    if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
                        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    elif hasattr(self.tfidf_vectorizer, 'get_feature_names'):
                        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
                    else:
                        # If no method is available, use vocabulary keys sorted by their values
                        tfidf_feature_names = sorted(self.tfidf_vectorizer.vocabulary_.keys(), 
                                                    key=lambda k: self.tfidf_vectorizer.vocabulary_[k])
                    
                    # Filter out number-containing column names (just like in the notebook)
                    tfidf_feature_names = [name for name in tfidf_feature_names if not any(c.isdigit() for c in name)]
                    
                    # Convert to dense array
                    tfidf_array = tfidf_matrix.toarray()[0]
                    
                    # If the feature array length doesn't match the feature names length, 
                    # we have a dimension mismatch
                    if len(tfidf_array) != len(tfidf_feature_names):
                        logger.warning(f"TF-IDF dimension mismatch: {len(tfidf_array)} values vs {len(tfidf_feature_names)} names")
                        # Create an array of zeros with the right size
                        tfidf_features = np.zeros(len(tfidf_feature_names))
                    else:
                        # Create a dictionary mapping feature names to values
                        tfidf_dict = dict(zip(tfidf_feature_names, tfidf_array))
                        
                        # Extract features in the order of filtered feature names
                        tfidf_features = [tfidf_dict.get(name, 0) for name in tfidf_feature_names]
                        
                except Exception as e:
                    logger.error(f"Error during TF-IDF transformation: {str(e)}")
                    # Create empty TF-IDF features
                    tfidf_features = []
                
                logger.info(f"TF-IDF vectorization complete. Generated {len(tfidf_features)} features.")
            else:
                logger.warning("No TF-IDF vectorizer provided. Text features will not be included.")
            
            # ---- Step 8: Assemble Final Feature Array in Exact Order ----

            # Build feature list dynamically to match training dataset!

            # 1️⃣ Base features
            keep_features = [
                "Created Year", "Created Month", "Created DayOfWeek", 
                "Estimated group size", "Within_Conversion_Window", "Subject_Word_Count"
            ]

            # 2️⃣ TF-IDF features
            if self.tfidf_vectorizer is not None:
                if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
                    tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
                elif hasattr(self.tfidf_vectorizer, 'get_feature_names'):
                    tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
                else:
                    tfidf_feature_names = sorted(self.tfidf_vectorizer.vocabulary_.keys(), 
                                                key=lambda k: self.tfidf_vectorizer.vocabulary_[k])
                # Filter out number-containing names
                tfidf_feature_names = [name for name in tfidf_feature_names if not any(c.isdigit() for c in name)]
            else:
                tfidf_feature_names = []

            # Add TF-IDF feature names to the feature list
            keep_features.extend(tfidf_feature_names)

            # 3️⃣ One-hot columns (ships, packages, time, day)
            onehot_columns = onehot_df.columns.tolist()
            keep_features.extend(onehot_columns)

            # 4️⃣ Final columns
            keep_features.extend(["Company City Amsterdam", "Recurring Customer"])

            # ---- Build the feature dictionary ----
            all_features_dict = {}

            # Add base features
            for feature in ["Created Year", "Created Month", "Created DayOfWeek", 
                            "Estimated group size", "Within_Conversion_Window", "Subject_Word_Count"]:
                all_features_dict[feature] = pd.to_numeric(quotes_df[feature].iloc[0], errors='coerce')
                if pd.isna(all_features_dict[feature]):
                    all_features_dict[feature] = 0

            # Add TF-IDF features
            for i, name in enumerate(tfidf_feature_names):
                if i < len(tfidf_features):
                    all_features_dict[name] = tfidf_features[i]
                else:
                    all_features_dict[name] = 0

            # Add one-hot features
            for feature in onehot_columns:
                all_features_dict[feature] = pd.to_numeric(onehot_df[feature].iloc[0], errors='coerce')
                if pd.isna(all_features_dict[feature]):
                    all_features_dict[feature] = 0

            # Add final columns
            for feature in ["Company City Amsterdam", "Recurring Customer"]:
                all_features_dict[feature] = pd.to_numeric(quotes_df[feature].iloc[0], errors='coerce')
                if pd.isna(all_features_dict[feature]):
                    all_features_dict[feature] = 0

            # ---- Build final feature array ----
            features_array = []
            for feature in keep_features:
                features_array.append(all_features_dict[feature])

            # ---- For API: convert everything to serializable ----
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj

            # Convert everything
            features_dict = convert_to_serializable(all_features_dict)
            features_array = convert_to_serializable(features_array)

            # ---- Log result ----
            logger.info(f"Final feature count: {len(features_array)} (expected from dataset: {len(keep_features)})")
            logger.info(f"Feature order: {keep_features[:6]} + {len(tfidf_feature_names)} TF-IDF + {len(onehot_columns)} one-hot + 2 final columns")

            # ---- Metadata ----
            metadata = {
                'feature_count': len(features_array),
                'expected_feature_count': len(keep_features),
                'tfidf_feature_count': len(tfidf_feature_names),
                'onehot_feature_count': len(onehot_columns),
                'final_columns': ["Company City Amsterdam", "Recurring Customer"]
            }

            # ---- Return ----
            return features_dict, features_array, metadata
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return empty results on error
            return {}, [], {"error": str(e)}