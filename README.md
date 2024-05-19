# Deep Learning VGG16 for simple vehichle classification

## Getting started

### Setting up the project

- Clone this repository: `git clone https://github.com/Mehrab-Evan/Audioshell_Backend_2024.git`
- Move the project directory: `cd AudioShell_Backend`
- Create the virtual environment and activate it:
- For windows: `python -m venv venv` & `.\venv\Scripts\activate`
- For Ubuntu or other Linux distributions: `python3 -m venv venv` & `source venv/bin/activate`
- Install the dependencies: `pip install -r requirements.txt`

### Project Dependencies

This project relies on several technologies and dependencies, including:

- [FastAPI](https://fastapi.tiangolo.com/): A modern, fast (high-performance), web framework for building APIs with Python.
- [SQLAlchemy](https://www.sqlalchemy.org/): A SQL toolkit and Object-Relational Mapping (ORM) library for Python.
- **PostgreSQL Database**: This project uses a PostgreSQL database to store and manage data.
- [Alembic](https://alembic.sqlalchemy.org/): A database migration tool for SQLAlchemy.
- ... (other project dependencies)

### Working on the project

- Move into the project directory: `cd AudioShell_Backend`
- Run the development task:
  - Setup the [Databases](#databases)
  - Start the server: `uvicorn main:app --reload`
  - Starts a server running at http://127.0.0.1:8000
  - Automatically restarts when any of your files change
  - Testing the api endpoints at http://127.0.0.1:8000/docs

### Databases

- Create a Database name: `audioshell`
- Run the following command to setup automatically the database:

```bash
alembic init alembic
```

- To generate a new Alembic migration:

```bash
alembic revision --autogenerate -m "YOUR MESSAGE"
```

- To apply the latest database migration:

```bash
alembic upgrade head
```
