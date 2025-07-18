from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app.main import bp
from app.models import User, DetectionResult, SupportRequest, Appointment
from app import db

@bp.route('/')
def index():
    """Home page"""
    # Get some statistics for the dashboard
    stats = {
        'total_users': User.query.count(),
        'total_detections': DetectionResult.query.count(),
        'total_support_requests': SupportRequest.query.count(),
        'total_appointments': Appointment.query.count()
    }
    
    # Get recent activity if user is logged in
    recent_activity = []
    if current_user.is_authenticated:
        recent_detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.timestamp.desc()).limit(5).all()
        recent_activity.extend([{
            'type': 'detection',
            'message': f"Brain scan analysis: {result.prediction_result}",
            'timestamp': result.timestamp
        } for result in recent_detections])
    
    return render_template('index.html', stats=stats, recent_activity=recent_activity)

@bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_stats = {}
    
    if current_user.user_type == 'patient':
        user_stats = {
            'detections': DetectionResult.query.filter_by(user_id=current_user.id).count(),
            'appointments': Appointment.query.filter_by(patient_id=current_user.id).count(),
            'support_requests': SupportRequest.query.filter_by(patient_id=current_user.id).count()
        }
        recent_detections = DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.timestamp.desc()).limit(5).all()
        recent_appointments = Appointment.query.filter_by(patient_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
        
    elif current_user.user_type == 'donor':
        user_stats = {
            'appointments': Appointment.query.filter_by(donor_id=current_user.id).count(),
            'support_responses': current_user.support_responses.count()
        }
        recent_detections = []
        recent_appointments = Appointment.query.filter_by(donor_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
        
    elif current_user.user_type == 'doctor':
        user_stats = {
            'appointments': Appointment.query.filter_by(doctor_id=current_user.id).count(),
            'patients': User.query.filter_by(user_type='patient').count()
        }
        recent_detections = DetectionResult.query.order_by(DetectionResult.timestamp.desc()).limit(5).all()
        recent_appointments = Appointment.query.filter_by(doctor_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         user_stats=user_stats, 
                         recent_detections=recent_detections,
                         recent_appointments=recent_appointments)

@bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@bp.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@bp.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    stats = {
        'total_users': User.query.count(),
        'patients': User.query.filter_by(user_type='patient').count(),
        'donors': User.query.filter_by(user_type='donor').count(),
        'doctors': User.query.filter_by(user_type='doctor').count(),
        'total_detections': DetectionResult.query.count(),
        'positive_detections': DetectionResult.query.filter_by(prediction_result='positive').count(),
        'negative_detections': DetectionResult.query.filter_by(prediction_result='negative').count(),
        'total_appointments': Appointment.query.count(),
        'pending_appointments': Appointment.query.filter_by(status='pending').count(),
        'confirmed_appointments': Appointment.query.filter_by(status='confirmed').count(),
        'total_support_requests': SupportRequest.query.count(),
        'open_support_requests': SupportRequest.query.filter_by(status='open').count()
    }
    return jsonify(stats)