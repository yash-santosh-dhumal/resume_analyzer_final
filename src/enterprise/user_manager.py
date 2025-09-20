"""
User Management System
Role-based access control for placement teams and students
"""

from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import secrets
import jwt
import logging
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles in the system"""
    STUDENT = "student"
    PLACEMENT_TEAM = "placement_team"
    LOCATION_ADMIN = "location_admin"
    SYSTEM_ADMIN = "system_admin"

class Permission(Enum):
    """System permissions"""
    # Resume permissions
    UPLOAD_RESUME = "upload_resume"
    VIEW_OWN_RESULTS = "view_own_results"
    
    # Job description permissions
    UPLOAD_JOB_DESCRIPTION = "upload_job_description"
    MANAGE_JOBS = "manage_jobs"
    
    # Analysis permissions
    RUN_SINGLE_ANALYSIS = "run_single_analysis"
    RUN_BULK_ANALYSIS = "run_bulk_analysis"
    VIEW_ALL_RESULTS = "view_all_results"
    EXPORT_RESULTS = "export_results"
    
    # Location permissions
    VIEW_LOCATION_ANALYTICS = "view_location_analytics"
    MANAGE_LOCATION_USERS = "manage_location_users"
    ACCESS_CROSS_LOCATION = "access_cross_location"
    
    # Admin permissions
    MANAGE_SYSTEM_CONFIG = "manage_system_config"
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_ALL_USERS = "manage_all_users"

@dataclass
class User:
    """User data model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    location: str
    full_name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserSession:
    """User session data"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

class UserManager:
    """
    Manages users, roles, permissions, and authentication
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize user manager"""
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = config.get("security", {}).get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.session_timeout = timedelta(hours=8)
        
        # In-memory storage (in production, use database)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, UserSession] = {}
        self.role_permissions = self._initialize_role_permissions()
        
        # Create default admin user
        self._create_default_admin()
        
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-based permissions"""
        return {
            UserRole.STUDENT: {
                Permission.UPLOAD_RESUME,
                Permission.VIEW_OWN_RESULTS,
                Permission.RUN_SINGLE_ANALYSIS
            },
            UserRole.PLACEMENT_TEAM: {
                Permission.UPLOAD_RESUME,
                Permission.UPLOAD_JOB_DESCRIPTION,
                Permission.MANAGE_JOBS,
                Permission.RUN_SINGLE_ANALYSIS,
                Permission.RUN_BULK_ANALYSIS,
                Permission.VIEW_ALL_RESULTS,
                Permission.EXPORT_RESULTS,
                Permission.VIEW_LOCATION_ANALYTICS,
                Permission.VIEW_OWN_RESULTS
            },
            UserRole.LOCATION_ADMIN: {
                Permission.UPLOAD_RESUME,
                Permission.UPLOAD_JOB_DESCRIPTION,
                Permission.MANAGE_JOBS,
                Permission.RUN_SINGLE_ANALYSIS,
                Permission.RUN_BULK_ANALYSIS,
                Permission.VIEW_ALL_RESULTS,
                Permission.EXPORT_RESULTS,
                Permission.VIEW_LOCATION_ANALYTICS,
                Permission.MANAGE_LOCATION_USERS,
                Permission.VIEW_OWN_RESULTS
            },
            UserRole.SYSTEM_ADMIN: set(Permission)  # All permissions
        }
    
    def _create_default_admin(self):
        """Create default system admin user"""
        admin_user = self.create_user(
            username="admin",
            email="admin@innomatics.in",
            password="admin123",  # Change in production
            role=UserRole.SYSTEM_ADMIN,
            location="hyderabad",
            full_name="System Administrator"
        )
        logger.info("Default admin user created")
    
    def create_user(self, username: str, email: str, password: str,
                   role: UserRole, location: str, full_name: str,
                   metadata: Optional[Dict[str, Any]] = None) -> User:
        """Create a new user"""
        # Validate username uniqueness
        if any(user.username == username for user in self.users.values()):
            raise ValueError("Username already exists")
        
        # Validate email uniqueness
        if any(user.email == email for user in self.users.values()):
            raise ValueError("Email already exists")
        
        # Generate user ID
        user_id = f"user_{secrets.token_urlsafe(8)}"
        
        # Hash password
        password_hash = self.pwd_context.hash(password)
        
        # Get role permissions
        permissions = self.role_permissions.get(role, set())
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            location=location,
            full_name=full_name,
            permissions=permissions.copy(),
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        logger.info(f"Created user {username} with role {role.value}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.pwd_context.verify(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        return user
    
    def create_session(self, user: User, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> UserSession:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.session_timeout
        
        session = UserSession(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if datetime.now() > session.expires_at:
            session.is_active = False
            return None
        
        return session if session.is_active else None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a user session"""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            return True
        return False
    
    def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "location": user.location,
            "permissions": [p.value for p in user.permissions],
            "exp": datetime.utcnow() + self.session_timeout,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Update allowed fields
        allowed_fields = {'email', 'full_name', 'location', 'is_active', 'metadata'}
        for field, value in updates.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)
        
        # Update password if provided
        if 'password' in updates:
            user.password_hash = self.pwd_context.hash(updates['password'])
        
        return True
    
    def change_user_role(self, user_id: str, new_role: UserRole, 
                        admin_user_id: str) -> bool:
        """Change user role (admin only)"""
        admin_user = self.get_user_by_id(admin_user_id)
        if not admin_user or admin_user.role not in [UserRole.SYSTEM_ADMIN, UserRole.LOCATION_ADMIN]:
            return False
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Location admin can only manage users in same location
        if admin_user.role == UserRole.LOCATION_ADMIN:
            if admin_user.location != user.location:
                return False
        
        user.role = new_role
        user.permissions = self.role_permissions.get(new_role, set()).copy()
        
        logger.info(f"Changed user {user.username} role to {new_role.value}")
        return True
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions
    
    def can_access_location(self, user: User, target_location: str) -> bool:
        """Check if user can access data from target location"""
        # System admin can access all locations
        if user.role == UserRole.SYSTEM_ADMIN:
            return True
        
        # Users can access their own location
        if user.location.lower() == target_location.lower():
            return True
        
        # Location admin can access cross-location with permission
        if user.role == UserRole.LOCATION_ADMIN:
            return Permission.ACCESS_CROSS_LOCATION in user.permissions
        
        # Placement team with cross-location permission
        if user.role == UserRole.PLACEMENT_TEAM:
            return Permission.ACCESS_CROSS_LOCATION in user.permissions
        
        return False
    
    def get_users_by_location(self, location: str, requester_user_id: str) -> List[User]:
        """Get all users in a specific location"""
        requester = self.get_user_by_id(requester_user_id)
        if not requester:
            return []
        
        # Check permissions
        if not (self.has_permission(requester, Permission.MANAGE_LOCATION_USERS) or
                self.has_permission(requester, Permission.MANAGE_ALL_USERS)):
            return []
        
        return [
            user for user in self.users.values()
            if user.location.lower() == location.lower()
        ]
    
    def get_users_by_role(self, role: UserRole, requester_user_id: str) -> List[User]:
        """Get all users with specific role"""
        requester = self.get_user_by_id(requester_user_id)
        if not requester:
            return []
        
        # Check permissions
        if not self.has_permission(requester, Permission.MANAGE_ALL_USERS):
            return []
        
        return [user for user in self.users.values() if user.role == role]
    
    def deactivate_user(self, user_id: str, admin_user_id: str) -> bool:
        """Deactivate a user account"""
        admin_user = self.get_user_by_id(admin_user_id)
        if not admin_user:
            return False
        
        # Check admin permissions
        if not (self.has_permission(admin_user, Permission.MANAGE_ALL_USERS) or
                self.has_permission(admin_user, Permission.MANAGE_LOCATION_USERS)):
            return False
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Location admin can only manage users in same location
        if admin_user.role == UserRole.LOCATION_ADMIN:
            if admin_user.location != user.location:
                return False
        
        user.is_active = False
        
        # Invalidate all user sessions
        for session in self.sessions.values():
            if session.user_id == user_id:
                session.is_active = False
        
        logger.info(f"Deactivated user {user.username}")
        return True
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics"""
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.is_active])
        
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = len([
                u for u in self.users.values() 
                if u.role == role and u.is_active
            ])
        
        location_counts = {}
        for user in self.users.values():
            if user.is_active:
                location_counts[user.location] = location_counts.get(user.location, 0) + 1
        
        active_sessions = len([s for s in self.sessions.values() if s.is_active])
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "role_distribution": role_counts,
            "location_distribution": location_counts,
            "active_sessions": active_sessions,
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            self.sessions[session_id].is_active = False
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_user_activity_log(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get user activity log (placeholder for future implementation)"""
        # In production, this would query an activity log database
        user = self.get_user_by_id(user_id)
        if not user:
            return []
        
        # Return mock activity data
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "login",
                "details": "User logged in successfully",
                "ip_address": "192.168.1.100"
            }
        ]