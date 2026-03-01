from flask import Blueprint, request, jsonify
from auth.models import db, User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json

    first_name = data.get('first_name')
    last_name  = data.get('last_name')
    email      = data.get('email')
    password   = data.get('password')

    if not first_name or not last_name or not email or not password:
        return jsonify({"message": "Missing required fields"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email already exists"}), 400

    new_user = User(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password=password
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User created successfully"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Missing credentials"}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"message": "User not found"}), 404

    if user.password != password:
        return jsonify({"message": "Invalid password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "firstName": user.first_name,
            "lastName": user.last_name,
            "email": user.email
        }
    })