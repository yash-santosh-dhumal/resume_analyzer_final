"""
Production Monitoring and Performance Optimization
Enterprise-grade monitoring, load balancing, and performance optimization
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import psutil
import threading
import time
import json
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SystemAlert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class LoadBalancerNode:
    """Load balancer node information"""
    node_id: str
    host: str
    port: int
    weight: int = 100
    active: bool = True
    current_load: int = 0
    max_load: int = 100
    health_score: float = 100.0
    last_health_check: datetime = field(default_factory=datetime.now)

class PerformanceMonitor:
    """
    Production performance monitoring system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitor"""
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)  # Last 10k metrics
        self.alerts = []
        self.alert_thresholds = self._load_alert_thresholds()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.collection_interval = config.get("monitoring", {}).get("interval_seconds", 30)
        
        # Performance baselines
        self.baselines = {}
        self.performance_history = defaultdict(list)
        
        logger.info("Performance monitor initialized")
    
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds from configuration"""
        return {
            "cpu_usage": {
                "warning": 70.0,
                "critical": 85.0
            },
            "memory_usage": {
                "warning": 75.0,
                "critical": 90.0
            },
            "disk_usage": {
                "warning": 80.0,
                "critical": 95.0
            },
            "response_time": {
                "warning": 2.0,  # seconds
                "critical": 5.0
            },
            "error_rate": {
                "warning": 2.0,  # percentage
                "critical": 5.0
            },
            "queue_length": {
                "warning": 50.0,
                "critical": 100.0
            }
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self._add_metric("cpu_usage", cpu_percent, "%", MetricType.SYSTEM)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric("memory_usage", memory.percent, "%", MetricType.SYSTEM)
        self._add_metric("memory_available", memory.available / (1024**3), "GB", MetricType.SYSTEM)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("disk_usage", disk_percent, "%", MetricType.SYSTEM)
        self._add_metric("disk_free", disk.free / (1024**3), "GB", MetricType.SYSTEM)
        
        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric("network_bytes_sent", network.bytes_sent, "bytes", MetricType.SYSTEM)
        self._add_metric("network_bytes_recv", network.bytes_recv, "bytes", MetricType.SYSTEM)
        
        # Process metrics
        process = psutil.Process()
        self._add_metric("process_cpu_percent", process.cpu_percent(), "%", MetricType.APPLICATION)
        self._add_metric("process_memory_mb", process.memory_info().rss / (1024**2), "MB", MetricType.APPLICATION)
        self._add_metric("process_threads", process.num_threads(), "count", MetricType.APPLICATION)
    
    def _add_metric(self, name: str, value: float, unit: str, metric_type: MetricType, tags: Dict[str, str] = None):
        """Add a metric to the buffer"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metric_type=metric_type,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
        
        # Update performance history
        self.performance_history[name].append((metric.timestamp, value))
        
        # Keep only last 1000 data points per metric
        if len(self.performance_history[name]) > 1000:
            self.performance_history[name] = self.performance_history[name][-1000:]
    
    def _check_alert_conditions(self):
        """Check if any metrics exceed alert thresholds"""
        recent_metrics = self._get_recent_metrics(minutes=5)
        
        for metric_name, thresholds in self.alert_thresholds.items():
            metrics = [m for m in recent_metrics if m.name == metric_name]
            if not metrics:
                continue
            
            # Use average of recent values
            avg_value = sum(m.value for m in metrics) / len(metrics)
            
            # Check critical threshold
            if avg_value >= thresholds.get("critical", float('inf')):
                self._create_alert(
                    metric_name, AlertLevel.CRITICAL, avg_value, thresholds["critical"]
                )
            # Check warning threshold
            elif avg_value >= thresholds.get("warning", float('inf')):
                self._create_alert(
                    metric_name, AlertLevel.WARNING, avg_value, thresholds["warning"]
                )
    
    def _create_alert(self, metric_name: str, level: AlertLevel, 
                     current_value: float, threshold_value: float):
        """Create a new alert"""
        import uuid
        
        # Check if similar alert already exists
        existing_alert = next(
            (alert for alert in self.alerts 
             if alert.metric_name == metric_name and alert.level == level and not alert.resolved),
            None
        )
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        alert = SystemAlert(
            alert_id=str(uuid.uuid4()),
            level=level,
            message=f"{metric_name} is {current_value:.2f}, exceeding {level.value} threshold of {threshold_value}",
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert.message}")
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up alerts older than 24 hours
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time or not alert.resolved
        ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        recent_metrics = self._get_recent_metrics(minutes=1)
        
        current = {}
        for metric in recent_metrics:
            if metric.name not in current or metric.timestamp > current[metric.name]["timestamp"]:
                current[metric.name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp
                }
        
        return current
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = self.performance_history.get(metric_name, [])
        recent_history = [
            {"timestamp": timestamp.isoformat(), "value": value}
            for timestamp, value in history
            if timestamp > cutoff_time
        ]
        
        return sorted(recent_history, key=lambda x: x["timestamp"])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        current_metrics = self.get_current_metrics()
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        # Calculate system health score
        health_score = self._calculate_health_score(current_metrics)
        
        return {
            "health_score": health_score,
            "status": self._determine_system_status(health_score),
            "current_metrics": current_metrics,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            "monitoring_active": self.monitoring_active,
            "last_update": datetime.now().isoformat()
        }
    
    def _get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [metric for metric in self.metrics_buffer if metric.timestamp > cutoff_time]
    
    def _calculate_health_score(self, current_metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # CPU health (100 - usage percentage)
        cpu_usage = current_metrics.get("cpu_usage", {}).get("value", 0)
        cpu_score = max(0, 100 - cpu_usage)
        scores.append(cpu_score)
        
        # Memory health
        memory_usage = current_metrics.get("memory_usage", {}).get("value", 0)
        memory_score = max(0, 100 - memory_usage)
        scores.append(memory_score)
        
        # Disk health
        disk_usage = current_metrics.get("disk_usage", {}).get("value", 0)
        disk_score = max(0, 100 - disk_usage)
        scores.append(disk_score)
        
        # Alert penalty
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        alert_penalty = len(active_alerts) * 5  # 5 points per active alert
        
        base_score = sum(scores) / len(scores) if scores else 100
        final_score = max(0, base_score - alert_penalty)
        
        return final_score
    
    def _determine_system_status(self, health_score: float) -> str:
        """Determine system status from health score"""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "warning"
        else:
            return "critical"
    
    def record_business_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record a business metric"""
        self._add_metric(name, value, unit, MetricType.BUSINESS, tags)
    
    def get_alerts(self, resolved: Optional[bool] = None) -> List[SystemAlert]:
        """Get system alerts"""
        if resolved is None:
            return self.alerts
        return [alert for alert in self.alerts if alert.resolved == resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

class LoadBalancer:
    """
    Simple load balancer for distributing analysis requests
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize load balancer"""
        self.config = config
        self.nodes = []
        self.current_node_index = 0
        self.health_check_interval = config.get("health_check_interval", 60)
        self.health_check_active = False
        
        # Load balancing strategies
        self.strategy = config.get("strategy", "round_robin")  # round_robin, least_connections, weighted
        
        logger.info("Load balancer initialized")
    
    def add_node(self, node_id: str, host: str, port: int, weight: int = 100):
        """Add a node to the load balancer"""
        node = LoadBalancerNode(
            node_id=node_id,
            host=host,
            port=port,
            weight=weight
        )
        self.nodes.append(node)
        logger.info(f"Added node {node_id} to load balancer")
    
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer"""
        self.nodes = [node for node in self.nodes if node.node_id != node_id]
        logger.info(f"Removed node {node_id} from load balancer")
    
    def get_next_node(self) -> Optional[LoadBalancerNode]:
        """Get the next node based on load balancing strategy"""
        active_nodes = [node for node in self.nodes if node.active and node.health_score > 50]
        
        if not active_nodes:
            logger.error("No active nodes available")
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(active_nodes)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(active_nodes)
        elif self.strategy == "weighted":
            return self._weighted_selection(active_nodes)
        else:
            return active_nodes[0]  # Default fallback
    
    def _round_robin_selection(self, nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Round-robin node selection"""
        node = nodes[self.current_node_index % len(nodes)]
        self.current_node_index = (self.current_node_index + 1) % len(nodes)
        return node
    
    def _least_connections_selection(self, nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Select node with least connections"""
        return min(nodes, key=lambda x: x.current_load)
    
    def _weighted_selection(self, nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Weighted node selection"""
        import random
        
        # Create weighted list
        weighted_nodes = []
        for node in nodes:
            weight = max(1, node.weight * (node.health_score / 100))
            weighted_nodes.extend([node] * int(weight))
        
        return random.choice(weighted_nodes) if weighted_nodes else nodes[0]
    
    def update_node_load(self, node_id: str, load_change: int):
        """Update node load"""
        for node in self.nodes:
            if node.node_id == node_id:
                node.current_load = max(0, node.current_load + load_change)
                break
    
    def check_node_health(self, node: LoadBalancerNode) -> float:
        """Check health of a specific node"""
        try:
            # In a real implementation, this would make an HTTP health check
            # For now, simulate health check based on load
            if node.current_load < node.max_load * 0.8:
                health_score = 100 - (node.current_load / node.max_load * 50)
            else:
                health_score = 50 - ((node.current_load - node.max_load * 0.8) / (node.max_load * 0.2) * 50)
            
            node.health_score = max(0, min(100, health_score))
            node.last_health_check = datetime.now()
            
            return node.health_score
            
        except Exception as e:
            logger.error(f"Health check failed for node {node.node_id}: {e}")
            node.health_score = 0
            return 0
    
    def start_health_checks(self):
        """Start periodic health checks"""
        if self.health_check_active:
            return
        
        self.health_check_active = True
        
        def health_check_loop():
            while self.health_check_active:
                try:
                    for node in self.nodes:
                        self.check_node_health(node)
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    time.sleep(5)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
        logger.info("Health checks started")
    
    def stop_health_checks(self):
        """Stop periodic health checks"""
        self.health_check_active = False
        logger.info("Health checks stopped")
    
    def get_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all nodes"""
        return [
            {
                "node_id": node.node_id,
                "host": node.host,
                "port": node.port,
                "active": node.active,
                "current_load": node.current_load,
                "max_load": node.max_load,
                "health_score": node.health_score,
                "last_health_check": node.last_health_check.isoformat()
            }
            for node in self.nodes
        ]

class PerformanceOptimizer:
    """
    Performance optimization engine
    """
    
    def __init__(self, monitor: PerformanceMonitor, config: Dict[str, Any]):
        """Initialize performance optimizer"""
        self.monitor = monitor
        self.config = config
        self.optimization_rules = self._load_optimization_rules()
        self.optimization_history = []
        
        logger.info("Performance optimizer initialized")
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load performance optimization rules"""
        return [
            {
                "name": "auto_scale_workers",
                "condition": lambda metrics: metrics.get("queue_length", {}).get("value", 0) > 20,
                "action": self._scale_workers,
                "description": "Automatically scale workers when queue is long"
            },
            {
                "name": "reduce_batch_size",
                "condition": lambda metrics: metrics.get("memory_usage", {}).get("value", 0) > 80,
                "action": self._reduce_batch_size,
                "description": "Reduce batch size when memory usage is high"
            },
            {
                "name": "cleanup_cache",
                "condition": lambda metrics: metrics.get("memory_usage", {}).get("value", 0) > 75,
                "action": self._cleanup_cache,
                "description": "Clean up cache when memory usage is high"
            }
        ]
    
    def run_optimization_cycle(self):
        """Run one optimization cycle"""
        current_metrics = self.monitor.get_current_metrics()
        
        optimizations_applied = []
        
        for rule in self.optimization_rules:
            try:
                if rule["condition"](current_metrics):
                    result = rule["action"](current_metrics)
                    if result:
                        optimizations_applied.append({
                            "rule": rule["name"],
                            "description": rule["description"],
                            "timestamp": datetime.now(),
                            "result": result
                        })
                        logger.info(f"Applied optimization: {rule['name']}")
            except Exception as e:
                logger.error(f"Error applying optimization rule {rule['name']}: {e}")
        
        if optimizations_applied:
            self.optimization_history.extend(optimizations_applied)
            # Keep only last 100 optimizations
            self.optimization_history = self.optimization_history[-100:]
        
        return optimizations_applied
    
    def _scale_workers(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Scale worker threads based on load"""
        queue_length = metrics.get("queue_length", {}).get("value", 0)
        
        if queue_length > 50:
            # In a real implementation, this would scale actual workers
            new_worker_count = min(20, queue_length // 10)
            return {
                "action": "scale_workers",
                "new_worker_count": new_worker_count,
                "reason": f"Queue length: {queue_length}"
            }
        
        return {}
    
    def _reduce_batch_size(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce batch size to save memory"""
        memory_usage = metrics.get("memory_usage", {}).get("value", 0)
        
        if memory_usage > 80:
            # Reduce batch size by 25%
            current_batch_size = self.config.get("bulk_processing", {}).get("batch_size", 50)
            new_batch_size = max(10, int(current_batch_size * 0.75))
            
            return {
                "action": "reduce_batch_size",
                "old_batch_size": current_batch_size,
                "new_batch_size": new_batch_size,
                "reason": f"Memory usage: {memory_usage}%"
            }
        
        return {}
    
    def _cleanup_cache(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up various caches"""
        memory_usage = metrics.get("memory_usage", {}).get("value", 0)
        
        if memory_usage > 75:
            # In a real implementation, this would clean actual caches
            return {
                "action": "cleanup_cache",
                "caches_cleaned": ["metrics_cache", "analysis_cache", "user_session_cache"],
                "reason": f"Memory usage: {memory_usage}%"
            }
        
        return {}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current metrics"""
        current_metrics = self.monitor.get_current_metrics()
        performance_summary = self.monitor.get_performance_summary()
        
        recommendations = []
        
        # CPU optimization
        cpu_usage = current_metrics.get("cpu_usage", {}).get("value", 0)
        if cpu_usage > 70:
            recommendations.append({
                "category": "CPU",
                "priority": "high" if cpu_usage > 85 else "medium",
                "recommendation": "Consider adding more CPU cores or optimizing CPU-intensive operations",
                "current_value": cpu_usage,
                "impact": "High"
            })
        
        # Memory optimization
        memory_usage = current_metrics.get("memory_usage", {}).get("value", 0)
        if memory_usage > 75:
            recommendations.append({
                "category": "Memory",
                "priority": "high" if memory_usage > 90 else "medium",
                "recommendation": "Implement memory optimization strategies or increase available RAM",
                "current_value": memory_usage,
                "impact": "High"
            })
        
        # System health
        health_score = performance_summary.get("health_score", 100)
        if health_score < 80:
            recommendations.append({
                "category": "System Health",
                "priority": "high" if health_score < 60 else "medium",
                "recommendation": "Address active alerts and optimize system performance",
                "current_value": health_score,
                "impact": "Critical"
            })
        
        return recommendations
    
    def get_optimization_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            opt for opt in self.optimization_history
            if opt["timestamp"] > cutoff_time
        ]

class ProductionManager:
    """
    Main production management orchestrator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize production manager"""
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.load_balancer = LoadBalancer(config.get("load_balancer", {}))
        self.optimizer = PerformanceOptimizer(self.monitor, config)
        
        # Management state
        self.running = False
        self.management_thread = None
        
        logger.info("Production manager initialized")
    
    def start(self):
        """Start production management"""
        if self.running:
            logger.warning("Production manager already running")
            return
        
        self.running = True
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start load balancer health checks
        self.load_balancer.start_health_checks()
        
        # Start management thread
        self.management_thread = threading.Thread(
            target=self._management_loop,
            name="ProductionManager",
            daemon=True
        )
        self.management_thread.start()
        
        logger.info("Production management started")
    
    def stop(self):
        """Stop production management"""
        self.running = False
        
        # Stop components
        self.monitor.stop_monitoring()
        self.load_balancer.stop_health_checks()
        
        if self.management_thread:
            self.management_thread.join(timeout=10)
        
        logger.info("Production management stopped")
    
    def _management_loop(self):
        """Main management loop"""
        optimization_interval = 300  # 5 minutes
        last_optimization = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Run optimization cycle periodically
                if (current_time - last_optimization).total_seconds() >= optimization_interval:
                    self.optimizer.run_optimization_cycle()
                    last_optimization = current_time
                
                # Brief sleep
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                time.sleep(10)
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "management_active": self.running,
            "performance": self.monitor.get_performance_summary(),
            "load_balancer": {
                "node_count": len(self.load_balancer.nodes),
                "active_nodes": len([n for n in self.load_balancer.nodes if n.active]),
                "nodes": self.load_balancer.get_node_status()
            },
            "optimization": {
                "recent_optimizations": len(self.optimizer.get_optimization_history(hours=1)),
                "recommendations": self.optimizer.get_optimization_recommendations()
            },
            "alerts": {
                "active": len(self.monitor.get_alerts(resolved=False)),
                "critical": len([a for a in self.monitor.get_alerts(resolved=False) if a.level == AlertLevel.CRITICAL])
            }
        }