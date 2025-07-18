import os
import base64
from flask import render_template, request, flash, redirect, url_for, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.detection import bp
from app.models import DetectionResult
from app.brain_tumor_detector import get_detector
from app import db

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/')
@login_required
def index():
    """Detection main page"""
    # Get user's detection history
    detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.timestamp.desc()).all()
    return render_template('detection/index.html', detections=detections)

@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload and analyze brain scan"""
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Get detector and make prediction
                detector = get_detector()
                result = detector.predict(filepath)
                
                # Save result to database
                detection_result = DetectionResult(
                    user_id=current_user.id,
                    image_filename=filename,
                    prediction_result=result.get('result', 'error'),
                    tumor_type=result.get('tumor_type'),
                    tumor_name=result.get('tumor_name'),
                    confidence_score=float(result.get('confidence', '0').replace('%', '')) if result.get('confidence') else None,
                    cnn_score=float(result.get('cnn_score', '0')) if result.get('cnn_score') else None,
                    matched_image=result.get('matched_image'),
                    distance=result.get('distance'),
                    notes=result.get('message', '')
                )
                
                db.session.add(detection_result)
                db.session.commit()
                
                # Encode image for display
                with open(filepath, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('detection/result.html', 
                                     result=result, 
                                     encoded_image=encoded_image,
                                     detection_id=detection_result.id)
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(request.url)
    
    return render_template('detection/upload.html')

@bp.route('/history')
@login_required
def history():
    """Detection history"""
    page = request.args.get('page', 1, type=int)
    detections = DetectionResult.query.filter_by(user_id=current_user.id)\
        .order_by(DetectionResult.timestamp.desc())\
        .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('detection/history.html', detections=detections)

@bp.route('/result/<int:detection_id>')
@login_required
def view_result(detection_id):
    """View specific detection result"""
    detection = DetectionResult.query.get_or_404(detection_id)
    
    # Check if user owns this detection or is a doctor
    if detection.user_id != current_user.id and current_user.user_type != 'doctor':
        flash('Access denied', 'error')
        return redirect(url_for('detection.history'))
    
    return render_template('detection/view_result.html', detection=detection)

@bp.route('/api/detect', methods=['POST'])
@login_required
def api_detect():
    """API endpoint for detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get detector and make prediction
        detector = get_detector()
        result = detector.predict(filepath)
        
        # Save result to database
        detection_result = DetectionResult(
            user_id=current_user.id,
            image_filename=filename,
            prediction_result=result.get('result', 'error'),
            tumor_type=result.get('tumor_type'),
            tumor_name=result.get('tumor_name'),
            confidence_score=float(result.get('confidence', '0').replace('%', '')) if result.get('confidence') else None,
            cnn_score=float(result.get('cnn_score', '0')) if result.get('cnn_score') else None,
            matched_image=result.get('matched_image'),
            distance=result.get('distance'),
            notes=result.get('message', '')
        )
        
        db.session.add(detection_result)
        db.session.commit()
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Add detection ID to result
        result['detection_id'] = detection_result.id
        
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@bp.route('/statistics')
@login_required
def statistics():
    """Detection statistics"""
    if current_user.user_type == 'doctor':
        # Doctors can see all statistics
        total_detections = DetectionResult.query.count()
        positive_detections = DetectionResult.query.filter_by(prediction_result='positive').count()
        negative_detections = DetectionResult.query.filter_by(prediction_result='negative').count()
        
        # Get tumor type distribution
        tumor_types = db.session.query(DetectionResult.tumor_type, db.func.count(DetectionResult.id))\
            .filter(DetectionResult.tumor_type.isnot(None))\
            .group_by(DetectionResult.tumor_type).all()
        
    else:
        # Patients can only see their own statistics
        total_detections = DetectionResult.query.filter_by(user_id=current_user.id).count()
        positive_detections = DetectionResult.query.filter_by(user_id=current_user.id, prediction_result='positive').count()
        negative_detections = DetectionResult.query.filter_by(user_id=current_user.id, prediction_result='negative').count()
        
        # Get user's tumor type distribution
        tumor_types = db.session.query(DetectionResult.tumor_type, db.func.count(DetectionResult.id))\
            .filter(DetectionResult.user_id == current_user.id)\
            .filter(DetectionResult.tumor_type.isnot(None))\
            .group_by(DetectionResult.tumor_type).all()
    
    stats = {
        'total_detections': total_detections,
        'positive_detections': positive_detections,
        'negative_detections': negative_detections,
        'tumor_types': dict(tumor_types)
    }
    
    return render_template('detection/statistics.html', stats=stats)