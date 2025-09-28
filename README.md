# TRANSITORY ARCHETYPE SEGMENTATION OF MASSACHUSETTS CENSUS BLOCK GROUPS FOR RETAIL STRATEGY
This project aims to segment each Census Block Group (CBG) in Massachusetts based on behavioral mobility patterns and transportation accessibility in order to recommend the most suitable retail store archetype for each region. An interactive streamlit app is built to render map and lets tune in feature weights live.
## Key Features:
- Transit proximity at scale: Uses GeoPandas to compute POI→nearest MBTA stop distances in meters (correct CRS handling)
- Robust multi-file ingestion: Reads and concatenates many SafeGraph CSVs in a folder; tolerant parsing with helpful logging and optional fallbacks
- Interactive UI: Clean, focused Streamlit app that renders the interactive CBG map
- Dynamic scoring: Live sliders to weight inputs (short-visit rate, visits/visitor, transit proximity, raw visits); weights auto-normalize and recompute the map instantly
## Requirements:
Python 3.10–3.11
## Setup Steps:
**1. Clone the repository**

`git clone https://github.com/KeerthikaK98/Transitory_Archetype_Segmentation`

**2. Setup tools**

Make sure Python is installed. To verify

Open command prompt

`python --version`

If it throws error, install Python and make sure to select 'Add .exe to PATH' during installation

**3. Install Requirements**

Install project dependencies

`pip install -r requirements.txt`

**4. Run the application**

Make sure utils.py, eda.py, and streamlit_app.py are run before deploying the Streamlit application. Use Visual Studio Code to run utils.py and eda.py

Once it is successful, open command prompt and run the below command:

`streamlit run streamlit_app.py`

The application will open in your default browser.



  


