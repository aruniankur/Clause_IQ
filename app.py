from flask import Flask, request, jsonify, render_template
from helper import get_qwen_response, getresult, embeddingmodel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

standard_template = {
    "TN" : "standard_tempate_default/TN_Standard_Template_Redacted.pdf",
    "WA" : "standard_tempate_default/WA_Standard_Redacted.pdf"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_attributes_file(file_path):
    """
    Process Excel or CSV file to extract attributes from the first column
    """
    try:
        # Get file extension
        file_ext = file_path.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            # Read CSV file
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            # Read Excel file
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Extract attributes from the first column
        if df.empty:
            raise ValueError("The file is empty")
        
        # Get the first column and remove any NaN values
        attributes = df.iloc[:, 0].dropna().astype(str).tolist()
        
        # Clean up attributes (remove extra whitespace)
        attributes = [attr.strip() for attr in attributes if attr.strip()]
        
        if not attributes:
            raise ValueError("No valid attributes found in the first column")
        
        return attributes, df.to_dict('list')
        
    except Exception as e:
        raise ValueError(f"Error processing attributes file: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get input from frontend
        workflow = request.form.get('workflow')
        print(f"\n=== ANALYZE ENDPOINT CALLED ===")
        print(f"Workflow: {workflow}")
        
        # Get attributes file
        attributes_file = request.files.get('attributesFile')
        if not attributes_file:
            return jsonify({'error': 'Attributes file is required'}), 400
        
        # Save the attributes file
        attributes_filename = secure_filename(attributes_file.filename or 'attributes.csv')
        attributes_path = os.path.join(app.config['UPLOAD_FOLDER'], attributes_filename)
        attributes_file.save(attributes_path)
        
        # Process the attributes file
        try:
            attributes , df_dict = process_attributes_file(attributes_path)
            print(f"\n=== ATTRIBUTES EXTRACTED ===")
            print(f"Number of attributes: {len(attributes)}")
            print(f"Attributes: {attributes}")

            # this need to be fixed
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        finally:
            # Clean up the temporary file
            if os.path.exists(attributes_path):
                os.remove(attributes_path)
        
        if not attributes:
            return jsonify({'error': 'No valid attributes found in the uploaded file'}), 400
        
        # Handle different workflows
        if workflow == '1':
            # Workflow 1: User uploads own template
            template_file = request.files.get('templateFile')
            personal_file = request.files.get('personalFile')
            
            if not template_file or not personal_file:
                return jsonify({'error': 'Both template and personal files are required'}), 400
            
            if not allowed_file(template_file.filename) or not allowed_file(personal_file.filename):
                return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX files are allowed for template and personal files'}), 400
            
            # Save uploaded files
            template_filename = secure_filename(template_file.filename or 'template.pdf')
            personal_filename = secure_filename(personal_file.filename or 'personal.pdf')
            
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
            personal_path = os.path.join(app.config['UPLOAD_FOLDER'], personal_filename)
            
            template_file.save(template_path)
            personal_file.save(personal_path)
            
            print(f"\n=== FILES SAVED ===")
            print(f"Template file: {template_filename}")
            print(f"Personal file: {personal_filename}")
            
        elif workflow == '2':
            # Workflow 2: Use template from dataset
            template_select = request.form.get('templateSelect')
            personal_file = request.files.get('personalFile')
            
            if not template_select or not personal_file:
                return jsonify({'error': 'Template selection and personal file are required'}), 400
            
            if not allowed_file(personal_file.filename):
                return jsonify({'error': 'Invalid file type. Only PDF, DOC, DOCX files are allowed for personal files'}), 400
            
            # Get template path from standard templates
            if template_select not in standard_template:
                return jsonify({'error': 'Invalid template selection'}), 400
            
            template_path = standard_template[template_select]
            personal_filename = secure_filename(personal_file.filename or 'personal.pdf')
            personal_path = os.path.join(app.config['UPLOAD_FOLDER'], personal_filename)
            personal_file.save(personal_path)
            
            print(f"\n=== FILES SAVED ===")
            # print(f"Template selection: {template_select}")
            # print(f"Template path: {template_path}")
            # print(f"Personal file: {personal_filename}")
            # print(f"DF Dictionary: {df_dict}")
        else:
            return jsonify({'error': 'Invalid workflow specified'}), 400
        
        # TODO: Add your processing logic here
        # You have access to:
        # - workflow: '1' or '2'
        # - attributes: list of attributes from CSV
        # - template_path: path to template file
        # - personal_path: path to personal file

        template_loader = PyPDFLoader(template_path)
        template_docs = template_loader.load()
        personal_loader = PyPDFLoader(personal_path)
        personal_docs = personal_loader.load()
        
        print("--------------------------------")
        print(f"Loaded {len(template_docs)} pages")
        print(f"Loaded {len(personal_docs)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,   # size of each chunk
        chunk_overlap=200, # overlap between chunks
        length_function=len,)

        template_docs_split = text_splitter.split_documents(template_docs)
        personal_docs_split = text_splitter.split_documents(personal_docs)
        template_docs_string = [i.page_content for i in template_docs_split]
        personal_docs_string = [i.page_content for i in personal_docs_split]
        template_embedding = embeddingmodel.encode(template_docs_string, convert_to_numpy=True)
        print(template_embedding.shape)
        personal_embedding = embeddingmodel.encode(personal_docs_string, convert_to_numpy=True)
        print(personal_embedding.shape)
        results = []
        for r in range(len(df_dict['Attribute'])):
            lst = df_dict.keys()
            all_attribute = ''
            for t in lst:
                all_attribute = all_attribute + df_dict[t][r] + ' '
            result = getresult(template_docs_string,template_embedding,personal_docs_string,personal_embedding,all_attribute,df_dict['Attribute'][r])
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        print(f"\n=== ERROR IN ANALYZE ENDPOINT ===")
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/process_excel_csv', methods=['POST'])
def process_excel_file():
    try:
        file = request.files['file']
        
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename or 'temp_file.csv')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"\n=== Processing file: {filename} ===")
        
        try:
            # Process the file based on extension
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext == 'csv':
                print("Reading CSV file...")
                df = pd.read_csv(file_path)
            elif file_ext in ['xlsx', 'xls']:
                print("Reading Excel file...")
                df = pd.read_excel(file_path)
            else:
                return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
            
            # Print file information to terminal
            # print(f"File shape: {df.shape}")
            # print(f"Columns: {list(df.columns)}")
            # print("\nFirst 10 rows:")
            # print(df.head(10).to_string())
            # print("\nData types:")
            # print(df.dtypes)
            
            # Convert DataFrame to dictionary for JSON response
            csv_table = df.to_dict('list')
            #print(f"\nConverted to dictionary: {csv_table}")
            
            return jsonify({
                'message': 'File processed successfully!',
                'csv_table': csv_table,
                'file_info': {
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'column_names': list(df.columns)
                }
            })
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400
            
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary file: {filename}")
                
    except Exception as e:
        print(f"Error in process_excel_file: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500



if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    
    app.run(host=host, port=port, debug=debug)