from flask import Flask, request, jsonify, render_template
from helper import get_embedding, get_qwen_response
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

standard_template = {
    "TN" : "standard_tempate_default/TN_Standard_Template_Redacted.pdf",
    "WA" : "standard_tempate_default/WA_Standard_Redacted.pdf"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        workflow = request.form.get('workflow')
        attributes = request.form.get('attributes', '').split(',')
        attributes = [attr.strip() for attr in attributes if attr.strip()]
        
        if workflow == '1':
            # Workflow 1: User uploads own template
            template_file = request.files.get('templateFile')
            personal_file = request.files.get('personalFile')
            
            if not template_file or not personal_file:
                return jsonify({'error': 'Both template and personal files are required'}), 400
            
            if not allowed_file(template_file.filename) or not allowed_file(personal_file.filename):
                return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX files are allowed'}), 400
            
            # Save uploaded files
            template_filename = secure_filename(template_file.filename or 'template.pdf')
            personal_filename = secure_filename(personal_file.filename or 'personal.pdf')
            
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
            personal_path = os.path.join(app.config['UPLOAD_FOLDER'], personal_filename)
            
            template_file.save(template_path)
            personal_file.save(personal_path)
            
            # Process files and generate comparison
            results = process_clause_comparison(template_path, personal_path, attributes)
            
        elif workflow == '2':
            # Workflow 2: Use template from dataset
            template_select = request.form.get('templateSelect')
            personal_file = request.files.get('personalFile')
            
            if not template_select or not personal_file:
                return jsonify({'error': 'Template selection and personal file are required'}), 400
            
            if not allowed_file(personal_file.filename):
                return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX files are allowed'}), 400
            
            # Get template path from standard templates
            if template_select not in standard_template:
                return jsonify({'error': 'Invalid template selection'}), 400
            
            template_path = standard_template[template_select]
            personal_filename = secure_filename(personal_file.filename or 'personal.pdf')
            personal_path = os.path.join(app.config['UPLOAD_FOLDER'], personal_filename)
            personal_file.save(personal_path)
            
            # Process files and generate comparison
            results = process_clause_comparison(template_path, personal_path, attributes)
            
        else:
            return jsonify({'error': 'Invalid workflow specified'}), 400
        
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def process_clause_comparison(template_path, personal_path, attributes):
    """
    Process the clause comparison between template and personal documents
    This is a placeholder function - you'll need to implement the actual AI processing
    """
    # For now, return sample data
    # In the actual implementation, you would:
    # 1. Load and parse both documents
    # 2. Extract relevant sections based on attributes
    # 3. Use AI to compare and analyze the clauses
    # 4. Return structured comparison results
    
    sample_results = []
    for i, attr in enumerate(attributes):
        sample_results.append({
            'attribute': attr,
            'your_clause': f'Sample clause content for {attr}',
            'template_clause': f'Template clause content for {attr}',
            'match_status': ['Good', 'Partial', 'Different'][i % 3],
            'confidence': f'{75 + (i * 5)}%',
            'similarity_score': 0.75 + (i * 0.05)
        })
    
    return sample_results

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    return jsonify({'message': 'File uploaded successfully!'})

if __name__ == '__main__':
    app.run(debug=True)