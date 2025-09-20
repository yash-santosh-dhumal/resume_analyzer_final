"""
Notification Service
Automated alerts and notifications for recruiters and students
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications"""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    WEBHOOK = "webhook"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class NotificationEvent(Enum):
    """Events that trigger notifications"""
    ANALYSIS_COMPLETED = "analysis_completed"
    BULK_JOB_COMPLETED = "bulk_job_completed"
    HIGH_SCORE_CANDIDATE = "high_score_candidate"
    SYSTEM_ALERT = "system_alert"
    JOB_REQUIREMENT_POSTED = "job_requirement_posted"
    FEEDBACK_REQUEST = "feedback_request"
    DEADLINE_REMINDER = "deadline_reminder"
    WEEKLY_REPORT = "weekly_report"

@dataclass
class NotificationTemplate:
    """Notification template"""
    event_type: NotificationEvent
    notification_type: NotificationType
    subject_template: str
    body_template: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    recipients_filter: Optional[Dict[str, Any]] = None

@dataclass
class Notification:
    """Individual notification"""
    notification_id: str
    event_type: NotificationEvent
    notification_type: NotificationType
    recipient: str
    subject: str
    content: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed, cancelled
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class NotificationChannel(ABC):
    """Abstract base class for notification channels"""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification through this channel"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is available"""
        pass

class EmailChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.use_tls = config.get("use_tls", True)
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            msg.attach(MIMEText(notification.content, 'html'))
            
            # Connect to server and send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.username, notification.recipient, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {notification.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {notification.recipient}: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if email channel is configured"""
        return bool(self.username and self.password)

class InAppChannel(NotificationChannel):
    """In-app notification channel"""
    
    def __init__(self):
        self.notifications_store = {}  # In production, use database
    
    async def send(self, notification: Notification) -> bool:
        """Store in-app notification"""
        try:
            user_notifications = self.notifications_store.get(notification.recipient, [])
            user_notifications.append({
                "id": notification.notification_id,
                "event_type": notification.event_type.value,
                "subject": notification.subject,
                "content": notification.content,
                "priority": notification.priority.value,
                "created_at": notification.created_at.isoformat(),
                "read": False
            })
            self.notifications_store[notification.recipient] = user_notifications
            
            logger.info(f"In-app notification stored for {notification.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in-app notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """In-app channel is always available"""
        return True
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        notifications = self.notifications_store.get(user_id, [])
        if unread_only:
            notifications = [n for n in notifications if not n.get("read", False)]
        return sorted(notifications, key=lambda x: x["created_at"], reverse=True)
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Mark notification as read"""
        notifications = self.notifications_store.get(user_id, [])
        for notification in notifications:
            if notification["id"] == notification_id:
                notification["read"] = True
                return True
        return False

class WebhookChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_urls = config.get("webhook_urls", {})
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp
            
            webhook_url = self.webhook_urls.get(notification.event_type.value)
            if not webhook_url:
                logger.warning(f"No webhook URL configured for {notification.event_type.value}")
                return False
            
            payload = {
                "event_type": notification.event_type.value,
                "recipient": notification.recipient,
                "subject": notification.subject,
                "content": notification.content,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat(),
                "metadata": notification.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully for {notification.event_type.value}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if webhook channel is configured"""
        return len(self.webhook_urls) > 0

class NotificationService:
    """
    Central notification service for the resume analysis system
    """
    
    def __init__(self, config: Dict[str, Any], user_manager):
        """Initialize notification service"""
        self.config = config
        self.user_manager = user_manager
        
        # Initialize notification channels
        self.channels = {
            NotificationType.EMAIL: EmailChannel(config.get("email", {})),
            NotificationType.IN_APP: InAppChannel(),
            NotificationType.WEBHOOK: WebhookChannel(config.get("webhook", {}))
        }
        
        # Notification templates
        self.templates = self._load_templates()
        
        # Notification queue and processing
        self.notification_queue = []
        self.processing = False
        
        # Subscription management
        self.subscriptions = self._load_default_subscriptions()
        
        logger.info("Notification service initialized")
    
    def _load_templates(self) -> Dict[NotificationEvent, Dict[NotificationType, NotificationTemplate]]:
        """Load notification templates"""
        templates = {}
        
        # Analysis completed template
        templates[NotificationEvent.ANALYSIS_COMPLETED] = {
            NotificationType.EMAIL: NotificationTemplate(
                event_type=NotificationEvent.ANALYSIS_COMPLETED,
                notification_type=NotificationType.EMAIL,
                subject_template="Resume Analysis Completed - {candidate_name}",
                body_template="""
                <h2>Resume Analysis Completed</h2>
                <p>Dear {recipient_name},</p>
                <p>The resume analysis for <strong>{candidate_name}</strong> has been completed.</p>
                <h3>Results Summary:</h3>
                <ul>
                    <li>Overall Score: <strong>{overall_score}%</strong></li>
                    <li>Match Level: <strong>{match_level}</strong></li>
                    <li>Recommendation: <strong>{hiring_decision}</strong></li>
                </ul>
                <p>You can view the detailed analysis in the dashboard.</p>
                <p>Best regards,<br>Innomatics Resume Analysis System</p>
                """,
                priority=NotificationPriority.NORMAL
            ),
            NotificationType.IN_APP: NotificationTemplate(
                event_type=NotificationEvent.ANALYSIS_COMPLETED,
                notification_type=NotificationType.IN_APP,
                subject_template="Analysis Complete: {candidate_name}",
                body_template="Resume analysis completed for {candidate_name}. Score: {overall_score}%. Recommendation: {hiring_decision}",
                priority=NotificationPriority.NORMAL
            )
        }
        
        # Bulk job completed template
        templates[NotificationEvent.BULK_JOB_COMPLETED] = {
            NotificationType.EMAIL: NotificationTemplate(
                event_type=NotificationEvent.BULK_JOB_COMPLETED,
                notification_type=NotificationType.EMAIL,
                subject_template="Bulk Analysis Job Completed - {job_id}",
                body_template="""
                <h2>Bulk Analysis Job Completed</h2>
                <p>Dear {recipient_name},</p>
                <p>Your bulk analysis job <strong>{job_id}</strong> has been completed.</p>
                <h3>Results Summary:</h3>
                <ul>
                    <li>Total Resumes: <strong>{total_resumes}</strong></li>
                    <li>Successfully Processed: <strong>{successful_resumes}</strong></li>
                    <li>Failed: <strong>{failed_resumes}</strong></li>
                    <li>HIRE Recommendations: <strong>{hire_count}</strong></li>
                    <li>INTERVIEW Recommendations: <strong>{interview_count}</strong></li>
                </ul>
                <p>You can download the detailed results from the dashboard.</p>
                <p>Best regards,<br>Innomatics Resume Analysis System</p>
                """,
                priority=NotificationPriority.HIGH
            )
        }
        
        # High score candidate template
        templates[NotificationEvent.HIGH_SCORE_CANDIDATE] = {
            NotificationType.EMAIL: NotificationTemplate(
                event_type=NotificationEvent.HIGH_SCORE_CANDIDATE,
                notification_type=NotificationType.EMAIL,
                subject_template="High Score Candidate Alert - {candidate_name}",
                body_template="""
                <h2>🌟 High Score Candidate Alert</h2>
                <p>Dear {recipient_name},</p>
                <p>We've identified a high-scoring candidate that matches your requirements:</p>
                <h3>Candidate Details:</h3>
                <ul>
                    <li>Name: <strong>{candidate_name}</strong></li>
                    <li>Score: <strong>{overall_score}%</strong></li>
                    <li>Location: <strong>{location}</strong></li>
                    <li>Key Skills: <strong>{top_skills}</strong></li>
                </ul>
                <p><strong>Recommendation: HIRE</strong></p>
                <p>This candidate scored in the top 10% for this position. Consider fast-tracking for interview.</p>
                <p>Best regards,<br>Innomatics Resume Analysis System</p>
                """,
                priority=NotificationPriority.HIGH
            ),
            NotificationType.IN_APP: NotificationTemplate(
                event_type=NotificationEvent.HIGH_SCORE_CANDIDATE,
                notification_type=NotificationType.IN_APP,
                subject_template="🌟 High Score Candidate: {candidate_name}",
                body_template="Exceptional candidate found! {candidate_name} scored {overall_score}% - Recommended for immediate interview.",
                priority=NotificationPriority.HIGH
            )
        }
        
        # System alert template
        templates[NotificationEvent.SYSTEM_ALERT] = {
            NotificationType.EMAIL: NotificationTemplate(
                event_type=NotificationEvent.SYSTEM_ALERT,
                notification_type=NotificationType.EMAIL,
                subject_template="System Alert - {alert_type}",
                body_template="""
                <h2>⚠️ System Alert</h2>
                <p>Dear Admin,</p>
                <p>A system alert has been triggered:</p>
                <h3>Alert Details:</h3>
                <ul>
                    <li>Type: <strong>{alert_type}</strong></li>
                    <li>Severity: <strong>{severity}</strong></li>
                    <li>Message: <strong>{message}</strong></li>
                    <li>Time: <strong>{timestamp}</strong></li>
                </ul>
                <p>Please check the system dashboard for more details.</p>
                <p>Best regards,<br>Innomatics Resume Analysis System</p>
                """,
                priority=NotificationPriority.URGENT
            )
        }
        
        # Weekly report template
        templates[NotificationEvent.WEEKLY_REPORT] = {
            NotificationType.EMAIL: NotificationTemplate(
                event_type=NotificationEvent.WEEKLY_REPORT,
                notification_type=NotificationType.EMAIL,
                subject_template="Weekly Analytics Report - {location}",
                body_template="""
                <h2>📊 Weekly Analytics Report</h2>
                <p>Dear {recipient_name},</p>
                <p>Here's your weekly analytics summary for <strong>{location}</strong>:</p>
                <h3>This Week's Performance:</h3>
                <ul>
                    <li>Total Analyses: <strong>{total_analyses}</strong></li>
                    <li>Average Score: <strong>{average_score}%</strong></li>
                    <li>Hire Rate: <strong>{hire_rate}%</strong></li>
                    <li>Top Skills: <strong>{top_skills}</strong></li>
                </ul>
                <h3>Trends:</h3>
                <ul>
                    <li>Volume Trend: <strong>{volume_trend}</strong></li>
                    <li>Quality Trend: <strong>{quality_trend}</strong></li>
                </ul>
                <p>View detailed analytics in the dashboard.</p>
                <p>Best regards,<br>Innomatics Resume Analysis System</p>
                """,
                priority=NotificationPriority.NORMAL
            )
        }
        
        return templates
    
    def _load_default_subscriptions(self) -> Dict[str, Dict[NotificationEvent, List[NotificationType]]]:
        """Load default notification subscriptions by role"""
        return {
            "student": {
                NotificationEvent.ANALYSIS_COMPLETED: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.FEEDBACK_REQUEST: [NotificationType.EMAIL, NotificationType.IN_APP]
            },
            "placement_team": {
                NotificationEvent.ANALYSIS_COMPLETED: [NotificationType.IN_APP],
                NotificationEvent.BULK_JOB_COMPLETED: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.HIGH_SCORE_CANDIDATE: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.WEEKLY_REPORT: [NotificationType.EMAIL]
            },
            "location_admin": {
                NotificationEvent.BULK_JOB_COMPLETED: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.HIGH_SCORE_CANDIDATE: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.SYSTEM_ALERT: [NotificationType.EMAIL, NotificationType.IN_APP],
                NotificationEvent.WEEKLY_REPORT: [NotificationType.EMAIL]
            },
            "system_admin": {
                NotificationEvent.SYSTEM_ALERT: [NotificationType.EMAIL, NotificationType.IN_APP, NotificationType.WEBHOOK],
                NotificationEvent.WEEKLY_REPORT: [NotificationType.EMAIL]
            }
        }
    
    async def send_notification(self, event_type: NotificationEvent, 
                               recipient_user_id: str, context: Dict[str, Any],
                               immediate: bool = False) -> bool:
        """Send notification for an event"""
        try:
            # Get recipient user
            user = self.user_manager.get_user_by_id(recipient_user_id)
            if not user or not user.is_active:
                logger.warning(f"User {recipient_user_id} not found or inactive")
                return False
            
            # Check user subscriptions
            user_subscriptions = self.subscriptions.get(user.role.value, {})
            notification_types = user_subscriptions.get(event_type, [])
            
            if not notification_types:
                logger.info(f"User {user.username} not subscribed to {event_type.value}")
                return True  # Not an error, just not subscribed
            
            # Create notifications for each subscribed type
            success = True
            for notification_type in notification_types:
                notification = self._create_notification(
                    event_type, notification_type, user, context
                )
                
                if immediate:
                    sent = await self._send_single_notification(notification)
                    success = success and sent
                else:
                    self.notification_queue.append(notification)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    def _create_notification(self, event_type: NotificationEvent, 
                           notification_type: NotificationType,
                           user, context: Dict[str, Any]) -> Notification:
        """Create notification from template"""
        import uuid
        
        # Get template
        template = self.templates.get(event_type, {}).get(notification_type)
        if not template:
            raise ValueError(f"No template found for {event_type.value}/{notification_type.value}")
        
        # Prepare template context
        template_context = {
            "recipient_name": user.full_name,
            "recipient_email": user.email,
            "location": user.location,
            **context
        }
        
        # Render template
        subject = template.subject_template.format(**template_context)
        content = template.body_template.format(**template_context)
        
        # Create notification
        return Notification(
            notification_id=str(uuid.uuid4()),
            event_type=event_type,
            notification_type=notification_type,
            recipient=user.email if notification_type == NotificationType.EMAIL else user.user_id,
            subject=subject,
            content=content,
            priority=template.priority,
            metadata=context
        )
    
    async def _send_single_notification(self, notification: Notification) -> bool:
        """Send a single notification"""
        try:
            channel = self.channels.get(notification.notification_type)
            if not channel or not channel.is_available():
                logger.warning(f"Channel {notification.notification_type.value} not available")
                return False
            
            success = await channel.send(notification)
            
            if success:
                notification.status = "sent"
                notification.sent_at = datetime.now()
                logger.info(f"Notification sent: {notification.notification_id}")
            else:
                notification.status = "failed"
                notification.retry_count += 1
                logger.error(f"Failed to send notification: {notification.notification_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification {notification.notification_id}: {e}")
            notification.status = "failed"
            notification.retry_count += 1
            return False
    
    async def process_notification_queue(self):
        """Process pending notifications in queue"""
        if self.processing or not self.notification_queue:
            return
        
        self.processing = True
        
        try:
            pending_notifications = self.notification_queue.copy()
            self.notification_queue.clear()
            
            for notification in pending_notifications:
                if notification.status == "pending":
                    await self._send_single_notification(notification)
                    # Small delay to avoid overwhelming channels
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing notification queue: {e}")
        
        finally:
            self.processing = False
    
    def subscribe_user(self, user_id: str, event_type: NotificationEvent, 
                      notification_types: List[NotificationType]) -> bool:
        """Subscribe user to notification event"""
        try:
            user = self.user_manager.get_user_by_id(user_id)
            if not user:
                return False
            
            role_subscriptions = self.subscriptions.get(user.role.value, {})
            role_subscriptions[event_type] = notification_types
            self.subscriptions[user.role.value] = role_subscriptions
            
            logger.info(f"Updated subscription for {user.username}: {event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update subscription: {e}")
            return False
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get in-app notifications for user"""
        in_app_channel = self.channels.get(NotificationType.IN_APP)
        if isinstance(in_app_channel, InAppChannel):
            return in_app_channel.get_user_notifications(user_id, unread_only)
        return []
    
    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark notification as read"""
        in_app_channel = self.channels.get(NotificationType.IN_APP)
        if isinstance(in_app_channel, InAppChannel):
            return in_app_channel.mark_as_read(user_id, notification_id)
        return False
    
    async def send_system_alert(self, alert_type: str, severity: str, 
                               message: str, context: Dict[str, Any] = None) -> bool:
        """Send system alert to administrators"""
        alert_context = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **(context or {})
        }
        
        # Send to all system admins
        admin_users = self.user_manager.get_users_by_role(
            self.user_manager.UserRole.SYSTEM_ADMIN, 
            "system"  # System admin user ID for permission check
        )
        
        success = True
        for admin in admin_users:
            sent = await self.send_notification(
                NotificationEvent.SYSTEM_ALERT,
                admin.user_id,
                alert_context,
                immediate=True  # System alerts are immediate
            )
            success = success and sent
        
        return success
    
    async def send_bulk_job_notification(self, job_id: str, job_summary: Dict[str, Any],
                                        submitter_user_id: str) -> bool:
        """Send bulk job completion notification"""
        context = {
            "job_id": job_id,
            "total_resumes": job_summary.get("total_resumes", 0),
            "successful_resumes": job_summary.get("processed_resumes", 0),
            "failed_resumes": job_summary.get("failed_resumes", 0),
            "hire_count": job_summary.get("result_statistics", {}).get("hire_recommendations", 0),
            "interview_count": job_summary.get("result_statistics", {}).get("interview_recommendations", 0)
        }
        
        return await self.send_notification(
            NotificationEvent.BULK_JOB_COMPLETED,
            submitter_user_id,
            context
        )
    
    async def send_high_score_alert(self, candidate_data: Dict[str, Any], 
                                   analysis_results: Dict[str, Any],
                                   location: str) -> bool:
        """Send high score candidate alert to placement team"""
        context = {
            "candidate_name": candidate_data.get("candidate_name", "Unknown"),
            "overall_score": analysis_results.get("overall_score", 0),
            "location": location,
            "top_skills": ", ".join(candidate_data.get("skills", [])[:5])
        }
        
        # Send to placement team in the same location
        placement_users = self.user_manager.get_users_by_location(location, "system")
        placement_team = [
            user for user in placement_users 
            if user.role.value in ["placement_team", "location_admin"]
        ]
        
        success = True
        for user in placement_team:
            sent = await self.send_notification(
                NotificationEvent.HIGH_SCORE_CANDIDATE,
                user.user_id,
                context,
                immediate=True  # High score alerts are immediate
            )
            success = success and sent
        
        return success
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification service statistics"""
        # Count notifications by status (simplified)
        return {
            "total_notifications_sent": 1000,  # Mock data
            "success_rate": 98.5,
            "average_delivery_time": 2.3,
            "active_subscriptions": len(self.subscriptions),
            "available_channels": [
                channel_type.value for channel_type, channel in self.channels.items()
                if channel.is_available()
            ],
            "queue_length": len(self.notification_queue)
        }