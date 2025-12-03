#!/bin/bash

# Market Pulse Engine (MPE) Startup Script
# Quick deployment and management for the Real-Time Market Intelligence System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Market Pulse Engine (MPE) - Quick Start Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all MPE services"
    echo "  stop        Stop all MPE services"
    echo "  restart     Restart all MPE services"
    echo "  status      Show service status"
    echo "  logs        Show service logs"
    echo "  dashboard   Open dashboard in browser"
    echo "  api-docs    Open API documentation"
    echo "  clean       Clean up all data and restart"
    echo "  dev         Start in development mode"
    echo "  health      Check system health"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start all services"
    echo "  $0 logs mpe-api       # Show API logs"
    echo "  $0 dashboard          # Open dashboard"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p logs
    mkdir -p data/postgres
    mkdir -p data/influxdb
    mkdir -p data/redis
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    print_success "Directories created"
}

# Function to start services
start_services() {
    print_status "Starting Market Pulse Engine services..."
    check_docker
    check_docker_compose
    setup_directories
    
    # Create environment file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# Market Pulse Engine Configuration
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)

# Database Configuration
POSTGRES_DB=mpe
POSTGRES_USER=mpe_user
POSTGRES_PASSWORD=mpe_password

# InfluxDB Configuration
INFLUXDB_ADMIN_TOKEN=mpe-token-$(openssl rand -base64 16)
INFLUXDB_ORG=mpe
INFLUXDB_BUCKET=market_data

# Development Mode
ENVIRONMENT=production
DEBUG=false

# External APIs (Optional - add your keys)
# ALPHA_VANTAGE_API_KEY=your-key-here
# FINNHUB_API_KEY=your-key-here
# FRED_API_KEY=your-key-here
EOF
        print_success ".env file created"
    fi
    
    # Start services
    docker-compose up -d
    
    print_success "Market Pulse Engine is starting..."
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be healthy
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    docker-compose ps
    
    print_success "✅ Market Pulse Engine is running!"
    print_status "🌐 Dashboard: http://localhost:3000"
    print_status "📊 API Docs: http://localhost:8000/docs"
    print_status "📈 Grafana: http://localhost:3001 (admin/admin)"
    print_status "📋 Prometheus: http://localhost:9090"
}

# Function to stop services
stop_services() {
    print_status "Stopping Market Pulse Engine services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting Market Pulse Engine services..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to show service status
show_status() {
    print_status "Market Pulse Engine Service Status:"
    docker-compose ps
    echo ""
    print_status "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
}

# Function to show logs
show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        print_status "Showing logs for all services..."
        docker-compose logs -f
    fi
}

# Function to open dashboard
open_dashboard() {
    if command -v open > /dev/null 2>&1; then
        # macOS
        open http://localhost:3000
    elif command -v xdg-open > /dev/null 2>&1; then
        # Linux
        xdg-open http://localhost:3000
    elif command -v start > /dev/null 2>&1; then
        # Windows
        start http://localhost:3000
    else
        print_status "Please open http://localhost:3000 in your browser"
    fi
}

# Function to open API docs
open_api_docs() {
    if command -v open > /dev/null 2>&1; then
        open http://localhost:8000/docs
    elif command -v xdg-open > /dev/null 2>&1; then
        xdg-open http://localhost:8000/docs
    elif command -v start > /dev/null 2>&1; then
        start http://localhost:8000/docs
    else
        print_status "Please open http://localhost:8000/docs in your browser"
    fi
}

# Function to clean and restart
clean_restart() {
    print_warning "This will remove all data and restart the system!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        rm -rf data/* logs/*
        print_success "Cleanup complete. Starting fresh..."
        start_services
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to start development mode
dev_mode() {
    print_status "Starting in development mode..."
    
    # Install Python dependencies if not in container
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Start database services only
    docker-compose up -d postgres influxdb redis
    
    print_status "Starting API server in development mode..."
    export DATABASE_URL="postgresql://mpe_user:mpe_password@localhost:5432/mpe"
    export INFLUXDB_URL="http://localhost:8086"
    export REDIS_URL="redis://localhost:6379/0"
    
    python main.py
}

# Function to check system health
check_health() {
    print_status "Checking Market Pulse Engine health..."
    
    # Check Docker services
    print_status "Checking Docker services..."
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker services are running"
    else
        print_warning "Some Docker services may not be running"
    fi
    
    # Check API health
    print_status "Checking API health..."
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API is responding"
    else
        print_error "API is not responding"
    fi
    
    # Check database connections
    print_status "Checking database connections..."
    if docker-compose exec postgres pg_isready -U mpe_user -d mpe > /dev/null; then
        print_success "PostgreSQL is ready"
    else
        print_warning "PostgreSQL may not be ready"
    fi
    
    if curl -s http://localhost:8086/ping > /dev/null; then
        print_success "InfluxDB is ready"
    else
        print_warning "InfluxDB may not be ready"
    fi
    
    if docker-compose exec redis redis-cli ping | grep -q PONG; then
        print_success "Redis is ready"
    else
        print_warning "Redis may not be ready"
    fi
    
    print_success "Health check complete"
}

# Function to show quick info
show_info() {
    echo "🚀 Market Pulse Engine (MPE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📊 Dashboard:    http://localhost:3000"
    echo "🔌 API Docs:     http://localhost:8000/docs"
    echo "📈 Grafana:      http://localhost:3001 (admin/admin)"
    echo "📋 Prometheus:   http://localhost:9090"
    echo ""
    echo "🔧 Commands:"
    echo "   ./start.sh start      - Start all services"
    echo "   ./start.sh dashboard  - Open dashboard"
    echo "   ./start.sh logs       - View logs"
    echo "   ./start.sh health     - Check health"
    echo "   ./start.sh stop       - Stop services"
    echo ""
    echo "💡 Quick Tips:"
    echo "   • Wait 2-3 minutes after starting for full initialization"
    echo "   • Check 'logs' command if services don't start"
    echo "   • Use 'clean' command to reset everything"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "dashboard")
        open_dashboard
        ;;
    "api-docs")
        open_api_docs
        ;;
    "clean")
        clean_restart
        ;;
    "dev")
        dev_mode
        ;;
    "health")
        check_health
        ;;
    "info")
        show_info
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac