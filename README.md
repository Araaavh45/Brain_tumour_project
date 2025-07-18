# Brain Tumor Project

This is an AI-powered brain tumor detection and patient support system built with Flask and deep learning models.

## Features
- Upload and analyze brain MRI scans
- Tumor detection and classification
- Patient support dashboard
- Detection history and results

## Project Structure
```
brain_tumor_project/
│
├── app/                        # Flask backend code
├── data/                       # Data, models, results
├── static/                     # CSS, images, etc.
├── templates/                  # HTML templates
├── instance/                   # Database and uploads
├── requirements.txt
├── README.md
├── run.py
└── .gitignore
```

## Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python run.py
   ```

## Notes
- Add your trained models to the `data/models/` folder
- Place your datasets in `data/datasets/`
- Sensitive files and large data are excluded via `.gitignore`

## License
MIT
