from datetime import datetime
from flask import render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from app.support import bp
from app.models import User, Message, Appointment, SupportRequest, SupportResponse
from app import db

@bp.route('/')
@login_required
def index():
    """Support system main page"""
    if current_user.user_type == 'patient':
        # Show patient support dashboard
        support_requests = SupportRequest.query.filter_by(patient_id=current_user.id).order_by(SupportRequest.created_at.desc()).limit(5).all()
        appointments = Appointment.query.filter_by(patient_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
        messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).limit(5).all()
        
    elif current_user.user_type == 'donor':
        # Show donor support dashboard
        support_requests = SupportRequest.query.filter_by(status='open').order_by(SupportRequest.created_at.desc()).limit(10).all()
        appointments = Appointment.query.filter_by(donor_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
        messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).limit(5).all()
        
    elif current_user.user_type == 'doctor':
        # Show doctor support dashboard
        support_requests = SupportRequest.query.order_by(SupportRequest.created_at.desc()).limit(10).all()
        appointments = Appointment.query.filter_by(doctor_id=current_user.id).order_by(Appointment.created_at.desc()).limit(5).all()
        messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).limit(5).all()
    
    return render_template('support/index.html', 
                         support_requests=support_requests,
                         appointments=appointments,
                         messages=messages)

@bp.route('/requests')
@login_required
def requests():
    """Support requests page"""
    page = request.args.get('page', 1, type=int)
    
    if current_user.user_type == 'patient':
        support_requests = SupportRequest.query.filter_by(patient_id=current_user.id)\
            .order_by(SupportRequest.created_at.desc())\
            .paginate(page=page, per_page=10, error_out=False)
    else:
        # Donors and doctors can see all requests
        support_requests = SupportRequest.query\
            .order_by(SupportRequest.created_at.desc())\
            .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('support/requests.html', support_requests=support_requests)

@bp.route('/request/new', methods=['GET', 'POST'])
@login_required
def new_request():
    """Create new support request"""
    if current_user.user_type != 'patient':
        flash('Only patients can create support requests', 'error')
        return redirect(url_for('support.index'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        support_type = request.form.get('support_type')
        urgency = request.form.get('urgency')
        
        support_request = SupportRequest(
            patient_id=current_user.id,
            title=title,
            description=description,
            support_type=support_type,
            urgency=urgency
        )
        
        db.session.add(support_request)
        db.session.commit()
        
        flash('Support request created successfully!', 'success')
        return redirect(url_for('support.view_request', request_id=support_request.id))
    
    return render_template('support/new_request.html')

@bp.route('/request/<int:request_id>')
@login_required
def view_request(request_id):
    """View support request details"""
    support_request = SupportRequest.query.get_or_404(request_id)
    responses = SupportResponse.query.filter_by(request_id=request_id).order_by(SupportResponse.created_at.asc()).all()
    
    return render_template('support/view_request.html', 
                         support_request=support_request,
                         responses=responses)

@bp.route('/request/<int:request_id>/respond', methods=['POST'])
@login_required
def respond_to_request(request_id):
    """Respond to support request"""
    support_request = SupportRequest.query.get_or_404(request_id)
    
    response_text = request.form.get('response_text')
    response_type = request.form.get('response_type', 'comment')
    
    response = SupportResponse(
        request_id=request_id,
        responder_id=current_user.id,
        response_text=response_text,
        response_type=response_type
    )
    
    db.session.add(response)
    
    # Update request status if needed
    if response_type == 'solution':
        support_request.status = 'resolved'
    elif support_request.status == 'open':
        support_request.status = 'in_progress'
    
    db.session.commit()
    
    flash('Response added successfully!', 'success')
    return redirect(url_for('support.view_request', request_id=request_id))

@bp.route('/messages')
@login_required
def messages():
    """Messages page"""
    page = request.args.get('page', 1, type=int)
    messages = Message.query.filter_by(recipient_id=current_user.id)\
        .order_by(Message.timestamp.desc())\
        .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('support/messages.html', messages=messages)

@bp.route('/message/new', methods=['GET', 'POST'])
@login_required
def new_message():
    """Send new message"""
    if request.method == 'POST':
        recipient_id = request.form.get('recipient_id')
        subject = request.form.get('subject')
        body = request.form.get('body')
        
        message = Message(
            sender_id=current_user.id,
            recipient_id=recipient_id,
            subject=subject,
            body=body
        )
        
        db.session.add(message)
        db.session.commit()
        
        flash('Message sent successfully!', 'success')
        return redirect(url_for('support.messages'))
    
    # Get potential recipients
    if current_user.user_type == 'patient':
        recipients = User.query.filter(User.user_type.in_(['donor', 'doctor'])).all()
    elif current_user.user_type == 'donor':
        recipients = User.query.filter(User.user_type.in_(['patient', 'doctor'])).all()
    else:  # doctor
        recipients = User.query.filter(User.user_type.in_(['patient', 'donor'])).all()
    
    return render_template('support/new_message.html', recipients=recipients)

@bp.route('/message/<int:message_id>')
@login_required
def view_message(message_id):
    """View message"""
    message = Message.query.get_or_404(message_id)
    
    # Check if user can view this message
    if message.recipient_id != current_user.id and message.sender_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('support.messages'))
    
    # Mark as read if recipient is viewing
    if message.recipient_id == current_user.id and not message.is_read:
        message.is_read = True
        db.session.commit()
    
    return render_template('support/view_message.html', message=message)

@bp.route('/appointments')
@login_required
def appointments():
    """Appointments page"""
    page = request.args.get('page', 1, type=int)
    
    if current_user.user_type == 'patient':
        appointments = Appointment.query.filter_by(patient_id=current_user.id)\
            .order_by(Appointment.appointment_date.desc())\
            .paginate(page=page, per_page=10, error_out=False)
    elif current_user.user_type == 'donor':
        appointments = Appointment.query.filter_by(donor_id=current_user.id)\
            .order_by(Appointment.appointment_date.desc())\
            .paginate(page=page, per_page=10, error_out=False)
    else:  # doctor
        appointments = Appointment.query.filter_by(doctor_id=current_user.id)\
            .order_by(Appointment.appointment_date.desc())\
            .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('support/appointments.html', appointments=appointments)

@bp.route('/appointment/new', methods=['GET', 'POST'])
@login_required
def new_appointment():
    """Schedule new appointment"""
    if request.method == 'POST':
        appointment_date = datetime.strptime(request.form.get('appointment_date'), '%Y-%m-%dT%H:%M')
        appointment_type = request.form.get('appointment_type')
        notes = request.form.get('notes')
        
        appointment = Appointment(
            patient_id=current_user.id if current_user.user_type == 'patient' else request.form.get('patient_id'),
            donor_id=request.form.get('donor_id') if request.form.get('donor_id') else None,
            doctor_id=request.form.get('doctor_id') if request.form.get('doctor_id') else None,
            appointment_date=appointment_date,
            appointment_type=appointment_type,
            notes=notes
        )
        
        db.session.add(appointment)
        db.session.commit()
        
        flash('Appointment scheduled successfully!', 'success')
        return redirect(url_for('support.view_appointment', appointment_id=appointment.id))
    
    # Get available doctors and donors
    doctors = User.query.filter_by(user_type='doctor').all()
    donors = User.query.filter_by(user_type='donor').all()
    patients = User.query.filter_by(user_type='patient').all() if current_user.user_type != 'patient' else []
    
    return render_template('support/new_appointment.html', 
                         doctors=doctors, 
                         donors=donors,
                         patients=patients)

@bp.route('/appointment/<int:appointment_id>')
@login_required
def view_appointment(appointment_id):
    """View appointment details"""
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check if user can view this appointment
    user_ids = [appointment.patient_id, appointment.donor_id, appointment.doctor_id]
    if current_user.id not in user_ids:
        flash('Access denied', 'error')
        return redirect(url_for('support.appointments'))
    
    return render_template('support/view_appointment.html', appointment=appointment)

@bp.route('/appointment/<int:appointment_id>/update', methods=['POST'])
@login_required
def update_appointment(appointment_id):
    """Update appointment status"""
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check if user can update this appointment
    user_ids = [appointment.patient_id, appointment.donor_id, appointment.doctor_id]
    if current_user.id not in user_ids:
        flash('Access denied', 'error')
        return redirect(url_for('support.appointments'))
    
    status = request.form.get('status')
    notes = request.form.get('notes')
    
    appointment.status = status
    if notes:
        appointment.notes = notes
    appointment.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    flash('Appointment updated successfully!', 'success')
    return redirect(url_for('support.view_appointment', appointment_id=appointment_id))

@bp.route('/donors')
@login_required
def donors():
    """List of available donors"""
    page = request.args.get('page', 1, type=int)
    donors = User.query.filter_by(user_type='donor', is_active=True)\
        .paginate(page=page, per_page=20, error_out=False)
    
    return render_template('support/donors.html', donors=donors)

@bp.route('/api/unread_messages')
@login_required
def api_unread_messages():
    """API endpoint for unread message count"""
    count = Message.query.filter_by(recipient_id=current_user.id, is_read=False).count()
    return jsonify({'count': count})