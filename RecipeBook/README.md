# RecipeBook - Culinary Recipe Publishing Site

RecipeBook is a simple web application for publishing and managing culinary recipes. Users can add, view, update, and delete recipes, including uploading images for each dish.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Setup and Installation

1.  **Clone the Repository (if applicable)**
    If you've downloaded this as a ZIP, extract it. If it's a git repo, clone it:
    ```bash
    git clone <repository_url>
    cd RecipeBook
    ```
    (If you are running this from a pre-existing project structure, you can skip this step and navigate to the `RecipeBook` directory.)

2.  **Create and Activate a Virtual Environment**
    It's highly recommended to use a virtual environment to manage project dependencies.

    On macOS and Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    With the virtual environment activated, install the required Python packages:
    ```bash
    pip install Flask Flask-SQLAlchemy Werkzeug
    ```

4.  **Initialize the Database (if running for the first time or `recipes.db` is missing)**
    The application uses a SQLite database (`instance/recipes.db`). The `app.py` script will attempt to create necessary folders like the upload folder on startup.

    If the `instance/recipes.db` file does not exist, you can create it and the necessary tables by running the following commands in a Python shell within the activated virtual environment in the `RecipeBook` directory:

    ```bash
    python
    ```
    Then, in the Python interpreter:
    ```python
    from app import app, db
    with app.app_context():
    db.create_all()
    exit()
    ```
    This will create the `recipes.db` file in an `instance` folder (Flask default) inside your `RecipeBook` directory. If the `instance` folder doesn't exist, Flask/SQLAlchemy should create it.

5.  **Ensure Upload Directory Exists**
    The application will attempt to create `RecipeBook/static/uploads/images/` on startup. If you encounter issues, ensure this directory path is writable by the application.

## Running the Application

1.  **Activate the Virtual Environment** (if not already active):
    ```bash
    # On macOS/Linux
    # source venv/bin/activate
    # On Windows
    # .\venv\Scripts\activate
    ```

2.  **Run the Flask Development Server**:
    Navigate to the `RecipeBook` directory (if you're not already there) and run:
    ```bash
    python app.py
    ```

3.  **Access the Application**:
    Open your web browser and go to: `http://127.0.0.1:5000/`

## Project Structure

-   `app.py`: Main Flask application file containing routes and logic.
-   `instance/recipes.db`: SQLite database file.
-   `static/`: Contains static assets:
    -   `css/`: Stylesheets.
    -   `img/`: General images for the site.
    -   `uploads/images/`: User-uploaded recipe images.
-   `templates/`: HTML templates for rendering pages.
-   `README.md`: This file.

## Features

-   View all recipes.
-   View detailed information for a single recipe.
-   Add new recipes with title, description, ingredients, steps, and an image.
-   Update existing recipes.
-   Delete recipes.
```
