import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import the classification functionality from our bot script
# This will also load the model into memory when the app starts
from classifier import TRASHNET_MODEL, classify_trash

app = Flask(__name__)

# Define a folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def index():
    """Renders the main page."""
    # We will create the index.html template in the next step
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_image_route():
    """Receives an image, classifies it, and returns the result."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file extension'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            # Get the classification result from our imported function
            trash_class = classify_trash(filepath, TRASHNET_MODEL)
            # Return the result as JSON
            return jsonify({'classification': trash_class})
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 503
        except Exception as e:
            # Log the error in a real app
            print(f"Error during classification: {e}")
            return jsonify({'error': 'Error during classification'}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'An unknown error occurred'}), 500


if __name__ == '__main__':
    # Using a different port to avoid potential conflicts
    app.run(debug=True, port=5001)
