#!/usr/bin/env python3
"""
Brain Tumor Detection and Patient Support System
Main application runner
"""

import os
from app import create_app, db
from app.models import User, DetectionResult, Message, Appointment, SupportRequest, SupportResponse

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'User': User,
        'DetectionResult': DetectionResult,
        'Message': Message,
        'Appointment': Appointment,
        'SupportRequest': SupportRequest,
        'SupportResponse': SupportResponse
    }

@app.cli.command()
def init_db():
    """Initialize the database with sample data"""
    db.create_all()
    
    # Create sample users
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@braintumor.com',
            first_name='System',
            last_name='Administrator',
            user_type='doctor',
            phone='1234567890'
        )
        admin.set_password('admin123')
        db.session.add(admin)
    
    if not User.query.filter_by(username='patient1').first():
        patient = User(
            username='patient1',
            email='patient1@example.com',
            first_name='John',
            last_name='Doe',
            user_type='patient',
            age=35,
            gender='Male',
            phone='9876543210',
            address='123 Main St, City, State',
            medical_history='No significant medical history'
        )
        patient.set_password('patient123')
        db.session.add(patient)
    
    if not User.query.filter_by(username='donor1').first():
        donor = User(
            username='donor1',
            email='donor1@example.com',
            first_name='Jane',
            last_name='Smith',
            user_type='donor',
            age=28,
            gender='Female',
            phone='5555555555',
            address='456 Oak Ave, City, State'
        )
        donor.set_password('donor123')
        db.session.add(donor)
    
    if not User.query.filter_by(username='doctor1').first():
        doctor = User(
            username='doctor1',
            email='doctor1@example.com',
            first_name='Dr. Sarah',
            last_name='Johnson',
            user_type='doctor',
            phone='7777777777',
            address='789 Medical Center, City, State'
        )
        doctor.set_password('doctor123')
        db.session.add(doctor)
    
    db.session.commit()
    print('Database initialized with sample data!')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)