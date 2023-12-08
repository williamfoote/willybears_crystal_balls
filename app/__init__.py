from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuration setup
    app.config.from_object('config.Config')

    # Initialize other components like database, login manager, etc.
    
    # Register Blueprints
    from app.routes import main
    app.register_blueprint(main)

    return app