#!/Users/ruterfu/miniconda3/bin/python
# -*- coding: utf-8 -*-

import hashlib
import json
import time
import secrets
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import redis
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

# FastAPI app
app = FastAPI(title="API Stats Query Service V2")

# Model pricing cache
MODEL_PRICING = {}
# Model display names mapping
MODEL_MAP = {}

# Configure CORS only in debug mode
DEBUG_MODE = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']

if DEBUG_MODE:
    print("🔧 DEBUG mode enabled - CORS configured for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
else:
    print("🔒 Production mode - CORS not configured")

# Redis connection - different hosts for debug/production
if DEBUG_MODE:
    REDIS_HOST = '127.0.0.1'
    REDIS_PORT = 16379
    print(f"📍 Using DEBUG Redis: {REDIS_HOST}:{REDIS_PORT}")
else:
    REDIS_HOST = os.environ.get('REDIS_HOST', '192.168.118.2')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
    print(f"📍 Using PRODUCTION Redis: {REDIS_HOST}:{REDIS_PORT}")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,  
    decode_responses=True,
    db=0
)


# Get SHOW_EXACT_TIME_LIST from environment
SHOW_EXACT_TIME_LIST = os.environ.get('SHOW_EXACT_TIME_LIST', '').split(',') if os.environ.get('SHOW_EXACT_TIME_LIST') else []
SHOW_EXACT_TIME_LIST = [user.strip() for user in SHOW_EXACT_TIME_LIST if user.strip()]
if SHOW_EXACT_TIME_LIST:
    print(f"📅 Exact time display enabled for users: {', '.join(SHOW_EXACT_TIME_LIST)}")

def format_last_used_at(last_used_at: str, user_name: str, requesting_user: str = None) -> str:
    """Format last_used_at timestamp based on user name and environment settings
    
    Args:
        last_used_at: The timestamp to format
        user_name: The name of the API key being displayed
        requesting_user: The name of the user making the request (for auth_token requests)
    """
    if not last_used_at:
        return ""
    
    # Check if user should see exact time
    # Priority: check requesting_user first (for auth_token requests), then user_name (for API key requests)
    check_user = requesting_user if requesting_user else user_name
    if check_user in SHOW_EXACT_TIME_LIST:
        try:
            dt = datetime.fromisoformat(last_used_at.replace('Z', '+00:00'))
            # Convert to UTC+8
            dt_utc8 = dt + timedelta(hours=8)
            return dt_utc8.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ""
    
    # Otherwise, format as relative time
    try:
        dt = datetime.fromisoformat(last_used_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - dt
        
        # Calculate time difference
        minutes = diff.total_seconds() / 60
        hours = minutes / 60
        days = hours / 24
        
        if minutes < 1:
            return "正在使用"
        elif hours < 1:
            return "刚刚使用"
        elif days < 1:
            return "一天内"
        elif days < 7:
            return "这个星期"
        elif days < 30:
            return "本月内"
        elif days < 60:
            return "两个月内"
        else:
            return "好久没用"
    except:
        return ""

# Salt for auth token generation
def load_auth_salt():
    """Load auth salt from file, generate if not exists"""
    salt_file = os.path.join(os.path.dirname(__file__), 'admin_salt.txt')
    
    try:
        if os.path.exists(salt_file):
            with open(salt_file, 'r', encoding='utf-8') as f:
                salt = f.read().strip()
                if salt:
                    print(f"✅ Loaded auth salt from {salt_file}")
                    return salt
        
        # Generate new random salt if file doesn't exist or is empty
        salt = secrets.token_urlsafe(32)
        with open(salt_file, 'w', encoding='utf-8') as f:
            f.write(salt)
        print(f"🔑 Generated new auth salt and saved to {salt_file}")
        return salt
        
    except Exception as e:
        print(f"⚠️ Failed to load/generate auth salt: {e}")
        return 'fallback_salt'

AUTH_TOKEN_SALT = load_auth_salt()

# Auth token cache (token -> token_info)
auth_token_cache = {}
MAX_AUTH_TOKENS = 100
AUTH_TOKEN_EXPIRY = 86400  # 24 hours in seconds
AUTH_TOKEN_FILE = 'auth_token.json'

# User messages cache (username -> message_data) - never expires unless Python restarts
user_messages_cache = {}

# Request log file
REQUEST_LOG_FILE = 'requests.log'

# Request models
class QueryRequest(BaseModel):
    api_key: Optional[str] = None
    auth_token: Optional[str] = None
    admin_secret: Optional[str] = None
    key_name: Optional[str] = None  # Add support for querying by name

# Response models  
class ModelStats(BaseModel):
    model: str
    requests: int
    inputTokens: int
    outputTokens: int
    allTokens: int
    cost: float

class ApiKeyStats(BaseModel):
    id: str
    name: str
    today_requests: int
    today_tokens: int
    today_cost: float
    today_input_tokens: int
    month_requests: int
    month_tokens: int
    month_cost: float
    month_input_tokens: int
    daily_cost_limit: float
    token_limit: int
    rate_limit_window: int
    rate_limit_requests: int
    concurrency_limit: int
    window_current_requests: int  # Current requests in the rate limit window
    window_current_tokens: int    # Current tokens in the rate limit window
    allowed_clients: List[str]
    model_stats_daily: List[ModelStats]
    model_stats_monthly: List[ModelStats]
    permissions: str
    is_active: bool
    created_at: str
    last_used_at: str

def get_date_in_timezone(date=None, offset=8):
    """Get date adjusted for timezone offset (default UTC+8)"""
    if date is None:
        date = datetime.now(timezone.utc)
    offset_delta = timedelta(hours=offset)
    adjusted_time = date + offset_delta
    return adjusted_time

def get_date_string_in_timezone(date=None, offset=8):
    """Get YYYY-MM-DD format date string in timezone"""
    tz_date = get_date_in_timezone(date, offset)
    return tz_date.strftime('%Y-%m-%d')

def get_current_month(date=None, offset=8):
    """Get YYYY-MM format month string in timezone"""
    tz_date = get_date_in_timezone(date, offset)
    return tz_date.strftime('%Y-%m')

def hash_api_key(api_key: str) -> str:
    """Hash API key using SHA256 with encryption key"""
    # Get encryption key from environment variable
    ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', '')
    if not ENCRYPTION_KEY:
        raise ValueError("ENCRYPTION_KEY environment variable is required")
    return hashlib.sha256((api_key + ENCRYPTION_KEY).encode()).hexdigest()

def generate_auth_token(api_key: str) -> str:
    """Generate auth token from API key using SHA1(MD5(api_key + salt))"""
    # First compute MD5(api_key + salt)
    md5_hash = hashlib.md5((api_key + AUTH_TOKEN_SALT).encode()).hexdigest()
    # Then compute SHA1 of the MD5 hash
    auth_token = hashlib.sha1(md5_hash.encode()).hexdigest()
    return auth_token

def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    # Priority 1: Try to get real IP from X-Real-IP header (set by proxy_set_header X-Real-IP $remote_addr;)
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip.strip()
    
    # Priority 2: Try X-Forwarded-For as fallback
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For may contain multiple IPs, get the first one
        return forwarded_for.split(',')[0].strip()
    
    # Fallback to client host
    return request.client.host if request.client else "unknown"

def mask_api_key(api_key: str) -> str:
    """Mask API key showing only first 5 and last 10 characters"""
    if not api_key or len(api_key) <= 15:
        return api_key  # Don't mask if too short
    
    # Show first 5 characters + ... + last 10 characters
    return f"{api_key[:5]}...{api_key[-10:]}"

def log_request(endpoint: str, client_ip: str, user_agent: str, auth_method: str, 
               masked_key: str, auth_token: str, user_id: str):
    """Log request details to requests.log file like nginx access.log"""
    try:
        # Format: [timestamp] IP "UA" endpoint auth_method masked_key auth_token user_id
        # Use UTC+8 (China timezone) for timestamp
        utc_now = datetime.now(timezone.utc)
        china_time = utc_now + timedelta(hours=8)
        timestamp = china_time.strftime('%d/%b/%Y:%H:%M:%S +0800')
        user_agent = user_agent.replace('"', '\\"')  # Escape quotes in UA
        
        log_entry = f'[{timestamp}] {client_ip} "{user_agent}" {endpoint} {auth_method} {masked_key} {auth_token[:16]}...{auth_token[-8:] if len(auth_token) > 24 else auth_token} {user_id}'
        
        log_file = os.path.join(os.path.dirname(__file__), REQUEST_LOG_FILE)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"⚠️ Failed to write request log: {e}")

def load_auth_tokens():
    """Load auth tokens from JSON file"""
    global auth_token_cache
    
    try:
        token_file = os.path.join(os.path.dirname(__file__), AUTH_TOKEN_FILE)
        if os.path.exists(token_file):
            with open(token_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Clean expired tokens and load valid ones
                current_time = time.time()
                valid_tokens = {}
                
                for token, info in data.items():
                    if current_time - info['created_at'] < AUTH_TOKEN_EXPIRY:
                        valid_tokens[token] = info
                
                auth_token_cache = valid_tokens
                print(f"✅ Loaded {len(valid_tokens)} valid auth tokens from {AUTH_TOKEN_FILE}")
                
                # Save back cleaned data if there were expired tokens
                if len(valid_tokens) < len(data):
                    save_auth_tokens()
                    print(f"🧹 Removed {len(data) - len(valid_tokens)} expired tokens")
        else:
            print(f"📄 Auth token file {AUTH_TOKEN_FILE} not found, starting with empty cache")
            
    except Exception as e:
        print(f"⚠️ Failed to load auth tokens: {e}")
        auth_token_cache = {}

def save_auth_tokens():
    """Save auth tokens to JSON file"""
    try:
        token_file = os.path.join(os.path.dirname(__file__), AUTH_TOKEN_FILE)
        with open(token_file, 'w', encoding='utf-8') as f:
            json.dump(auth_token_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Failed to save auth tokens: {e}")

def validate_auth_token(auth_token: str, request: Request = None) -> bool:
    """Validate auth token from cache and verify user_id still exists in Redis"""
    global auth_token_cache
    
    # Check if token exists in cache
    if auth_token not in auth_token_cache:
        return False
    
    token_info = auth_token_cache[auth_token]
    
    # Check if token is expired
    if time.time() - token_info['created_at'] > AUTH_TOKEN_EXPIRY:
        # Remove expired token
        del auth_token_cache[auth_token]
        save_auth_tokens()
        return False
    
    # Check if user_id still exists in Redis (if user_id is stored)
    user_id = token_info.get('user_id')
    if user_id:
        try:
            # Check if the API key still exists and is active in Redis
            key_data = redis_client.hgetall(f'apikey:{user_id}')
            if not key_data or key_data.get('isActive') != 'true':
                # User ID no longer exists or is inactive, remove token
                del auth_token_cache[auth_token]
                save_auth_tokens()
                print(f"🔒 Auth token invalidated: user_id {user_id} no longer exists or inactive")
                return False
        except Exception as e:
            print(f"⚠️ Error checking user_id {user_id} in Redis: {e}")
            # If Redis check fails, invalidate the token for security
            del auth_token_cache[auth_token]
            save_auth_tokens()
            return False
    
    # Update last access time, IP and user agent if request is provided
    if request:
        token_info['last_access'] = time.time()
        token_info['last_ip'] = get_client_ip(request)
        token_info['last_user_agent'] = request.headers.get('User-Agent', 'Unknown')
        save_auth_tokens()
    
    return True

def add_auth_token(auth_token: str, request: Request, user_id: str = None, user_name: str = None):
    """Add auth token to cache with user info and save to file"""
    global auth_token_cache
    
    # If cache is full, remove oldest tokens
    if len(auth_token_cache) >= MAX_AUTH_TOKENS:
        # Sort by created_at and remove oldest half
        sorted_tokens = sorted(auth_token_cache.items(), key=lambda x: x[1]['created_at'])
        tokens_to_remove = sorted_tokens[:len(sorted_tokens)//2]
        for token, _ in tokens_to_remove:
            del auth_token_cache[token]
    
    # Get client IP and user agent
    client_ip = get_client_ip(request)
    user_agent = request.headers.get('User-Agent', 'Unknown')
    current_time = time.time()
    
    # Add new token with detailed info including user_id and user_name
    auth_token_cache[auth_token] = {
        'created_at': current_time,
        'last_access': current_time,
        'ip_address': client_ip,
        'last_ip': client_ip,
        'user_agent': user_agent,
        'last_user_agent': user_agent,
        'user_id': user_id,  # Store the user_id for validation
        'user_name': user_name  # Store the user_name for display
    }
    
    # Save to file
    save_auth_tokens()
    print(f"🔐 New auth token created for user: {user_name or user_id} | IP: {client_ip} | User-Agent: {user_agent[:50]}{'...' if len(user_agent) > 50 else ''}")

def validate_api_key(api_key: str) -> Optional[str]:
    """Validate API key and return its ID"""
    if not api_key or not api_key.startswith('cr_'):
        return None
    
    # Hash the API key
    hashed_key = hash_api_key(api_key)
    
    # Find key ID by hash
    key_id = redis_client.hget('apikey:hash_map', hashed_key)
    if not key_id:
        return None
    
    # Check if key exists and is active
    key_data = redis_client.hgetall(f'apikey:{key_id}')
    if not key_data:
        return None
    
    # Check if active
    if key_data.get('isActive') != 'true':
        return None
    
    # Check expiration
    expires_at = key_data.get('expiresAt')
    if expires_at and datetime.fromisoformat(expires_at.replace('Z', '+00:00')) < datetime.now(timezone.utc):
        return None
    
    return key_id

def get_api_key_stats(key_id: str, include_sensitive: bool = False, requesting_user: str = None) -> Dict[str, Any]:
    """Get statistics for a single API key"""
    today = get_date_string_in_timezone()
    current_month = get_current_month()
    
    # Get API key data
    key_data = redis_client.hgetall(f'apikey:{key_id}')
    if not key_data:
        return None
    
    # Skip inactive keys
    if key_data.get('isActive') != 'true':
        return None
    
    # Get daily stats
    daily_stats = redis_client.hgetall(f'usage:daily:{key_id}:{today}')
    monthly_stats = redis_client.hgetall(f'usage:monthly:{key_id}:{current_month}')
    
    # Get costs
    daily_cost = float(redis_client.get(f'usage:cost:daily:{key_id}:{today}') or 0)
    monthly_cost = float(redis_client.get(f'usage:cost:monthly:{key_id}:{current_month}') or 0)
    
    # Parse allowed clients
    allowed_clients = []
    try:
        allowed_clients = json.loads(key_data.get('allowedClients', '[]'))
    except:
        pass
    
    # Get model stats - daily
    model_stats_daily = []
    daily_model_keys = redis_client.keys(f'usage:{key_id}:model:daily:*:{today}')
    for key in daily_model_keys:
        parts = key.split(':')
        if len(parts) >= 6:
            model = parts[4]
            model_data = redis_client.hgetall(key)
            if model_data:
                input_tokens = int(model_data.get('inputTokens', 0))
                output_tokens = int(model_data.get('outputTokens', 0))
                cache_read_tokens = int(model_data.get('cacheReadTokens', 0))
                cache_create_tokens = int(model_data.get('cacheCreateTokens', 0))
                
                # Calculate cost based on model pricing
                cost = 0.0
                if model in MODEL_PRICING:
                    pricing = MODEL_PRICING[model]
                    input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                    output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                    cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                    cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                    cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                
                model_stats_daily.append({
                    'model': model,
                    'requests': int(model_data.get('requests', 0)),
                    'inputTokens': input_tokens,
                    'outputTokens': output_tokens,
                    'cacheReadTokens': cache_read_tokens,
                    'cacheCreateTokens': cache_create_tokens,
                    'allTokens': int(model_data.get('allTokens', 0)),
                    'cost': cost
                })
    
    # Get model stats - monthly
    model_stats_monthly = []
    monthly_model_keys = redis_client.keys(f'usage:{key_id}:model:monthly:*:{current_month}')
    for key in monthly_model_keys:
        parts = key.split(':')
        if len(parts) >= 6:
            model = parts[4]
            model_data = redis_client.hgetall(key)
            if model_data:
                input_tokens = int(model_data.get('inputTokens', 0))
                output_tokens = int(model_data.get('outputTokens', 0))
                cache_read_tokens = int(model_data.get('cacheReadTokens', 0))
                cache_create_tokens = int(model_data.get('cacheCreateTokens', 0))
                
                # Calculate cost based on model pricing
                cost = 0.0
                if model in MODEL_PRICING:
                    pricing = MODEL_PRICING[model]
                    input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                    output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                    cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                    cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                    cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                
                model_stats_monthly.append({
                    'model': model,
                    'requests': int(model_data.get('requests', 0)),
                    'inputTokens': input_tokens,
                    'outputTokens': output_tokens,
                    'cacheReadTokens': cache_read_tokens,
                    'cacheCreateTokens': cache_create_tokens,
                    'allTokens': int(model_data.get('allTokens', 0)),
                    'cost': cost
                })
    
    # Get current window usage and session window info
    window_current_requests = 0
    window_current_tokens = 0
    session_window_start = None
    session_window_end = None
    
    # Check if API key has a rate limit window
    rate_limit_window = int(key_data.get('rateLimitWindow', 0))
    
    # Also check if API key is associated with a Claude account
    claude_account_id = key_data.get('claudeAccountId', '')
    
    if claude_account_id:
        # Get Claude account data to check session window
        account_data = redis_client.hgetall(f'claude:account:{claude_account_id}')
        if account_data:
            session_window_start = account_data.get('sessionWindowStart')
            session_window_end = account_data.get('sessionWindowEnd')
            
            # If there's an active session window, calculate usage within it
            if session_window_start and session_window_end:
                now = datetime.now(timezone.utc)
                window_end = datetime.fromisoformat(session_window_end.replace('Z', '+00:00'))
                window_start = datetime.fromisoformat(session_window_start.replace('Z', '+00:00'))
                
                # Check if we're within the session window
                if window_start <= now <= window_end:
                    # Calculate hours since window start
                    hours_since_start = int((now - window_start).total_seconds() / 3600)
                    
                    # Sum up hourly usage within the window
                    window_requests = 0
                    window_tokens = 0
                    
                    for hour_offset in range(hours_since_start + 1):
                        hour_time = window_start + timedelta(hours=hour_offset)
                        hour_key = f"account_usage:hourly:{claude_account_id}:{hour_time.strftime('%Y-%m-%d')}:{hour_time.hour:02d}"
                        hour_data = redis_client.hgetall(hour_key)
                        if hour_data:
                            window_requests += int(hour_data.get('requests', 0))
                            window_tokens += int(hour_data.get('allTokens', 0) or hour_data.get('tokens', 0))
                    
                    # Use the window usage if we have rate limits
                    if rate_limit_window > 0:
                        window_current_requests = window_requests
                        window_current_tokens = window_tokens
    
    # Fallback: if no window data found but has rate limits, estimate from daily usage
    if rate_limit_window > 0 and window_current_requests == 0 and window_current_tokens == 0:
        window_current_requests = min(int(daily_stats.get('requests', 0)), int(key_data.get('rateLimitRequests', 0)))
        window_current_tokens = min(int(daily_stats.get('allTokens', 0)), int(key_data.get('tokenLimit', 0)))
    
    # Calculate window cost for this API key if there's an active window
    window_cost = 0.0
    window_info = None
    
    if claude_account_id and session_window_start and session_window_end:
        try:
            window_start = datetime.fromisoformat(session_window_start.replace('Z', '+00:00'))
            window_end = datetime.fromisoformat(session_window_end.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            
            # Only calculate if window is active
            if window_start <= now <= window_end:
                window_cost = calculate_window_cost_for_key(key_id, window_start, window_end)
                
                # Convert to UTC+8 for display
                window_start_tz8 = window_start + timedelta(hours=8)
                window_end_tz8 = window_end + timedelta(hours=8)
                
                window_info = {
                    'start_hour': window_start_tz8.hour,
                    'start_min': window_start_tz8.minute,
                    'end_hour': window_end_tz8.hour,
                    'end_min': window_end_tz8.minute,
                    'cost': window_cost
                }
        except Exception as e:
            print(f"Error calculating window cost for key {key_id}: {e}")
    
    # Build response with or without sensitive data
    result = {
        'name': key_data.get('name', 'Unknown'),
        'today_requests': int(daily_stats.get('requests', 0)),
        'today_tokens': int(daily_stats.get('allTokens', 0) or daily_stats.get('tokens', 0)),
        'today_cost': daily_cost,
        'today_input_tokens': int(daily_stats.get('inputTokens', 0)),
        'month_requests': int(monthly_stats.get('requests', 0)),
        'month_tokens': int(monthly_stats.get('allTokens', 0) or monthly_stats.get('tokens', 0)),
        'month_cost': monthly_cost,
        'month_input_tokens': int(monthly_stats.get('inputTokens', 0)),
        'daily_cost_limit': float(key_data.get('dailyCostLimit', 0)),
        'token_limit': int(key_data.get('tokenLimit', 0)),
        'rate_limit_window': int(key_data.get('rateLimitWindow', 0)),
        'rate_limit_requests': int(key_data.get('rateLimitRequests', 0)),
        'concurrency_limit': int(key_data.get('concurrencyLimit', 0)),
        'window_current_requests': window_current_requests,
        'window_current_tokens': window_current_tokens,
        'window_info': window_info,  # Add window information
        'model_stats_daily': model_stats_daily,
        'model_stats_monthly': model_stats_monthly,
        'permissions': key_data.get('permissions', 'unknown'),
        'is_active': key_data.get('isActive') == 'true',
        'last_used_at': format_last_used_at(key_data.get('lastUsedAt', ''), key_data.get('name', 'Unknown'), requesting_user),
        '_raw_last_used_at': key_data.get('lastUsedAt', ''),  # Add raw timestamp for sorting
        'status': None  # Will be calculated later when accounts info is available
    }
    
    # Include sensitive data only if requested
    if include_sensitive:
        result['id'] = key_id
        result['allowed_clients'] = allowed_clients
        result['created_at'] = key_data.get('createdAt', '')
    
    return result

def get_api_key_stats_from_data(key_id: str, all_redis_data: Dict[str, Any], include_sensitive: bool = False, requesting_user: str = None) -> Dict[str, Any]:
    """Get statistics for a single API key from preloaded Redis data"""
    today = get_date_string_in_timezone()
    current_month = get_current_month()
    
    # Get API key data from preloaded data
    key_data_key = f'apikey:{key_id}'
    if key_data_key not in all_redis_data:
        return None
        
    key_data = all_redis_data[key_data_key]
    if not key_data:
        return None
    
    # Skip inactive keys
    if key_data.get('isActive') != 'true':
        return None
    
    # Get daily and monthly stats from preloaded data
    daily_stats_key = f'usage:daily:{key_id}:{today}'
    monthly_stats_key = f'usage:monthly:{key_id}:{current_month}'
    daily_cost_key = f'usage:cost:daily:{key_id}:{today}'
    monthly_cost_key = f'usage:cost:monthly:{key_id}:{current_month}'
    
    daily_stats = all_redis_data.get(daily_stats_key, {})
    monthly_stats = all_redis_data.get(monthly_stats_key, {})
    
    # Get costs - these are string values in Redis
    daily_cost = 0.0
    monthly_cost = 0.0
    
    # Handle cost keys which might be stored as single values
    if daily_cost_key in all_redis_data:
        try:
            cost_data = all_redis_data[daily_cost_key]
            if isinstance(cost_data, str):
                daily_cost = float(cost_data)
            elif isinstance(cost_data, dict) and 'cost' in cost_data:
                daily_cost = float(cost_data['cost'])
        except (ValueError, TypeError):
            daily_cost = 0.0
    
    if monthly_cost_key in all_redis_data:
        try:
            cost_data = all_redis_data[monthly_cost_key]
            if isinstance(cost_data, str):
                monthly_cost = float(cost_data)
            elif isinstance(cost_data, dict) and 'cost' in cost_data:
                monthly_cost = float(cost_data['cost'])
        except (ValueError, TypeError):
            monthly_cost = 0.0
    
    # Parse allowed clients
    allowed_clients = []
    try:
        allowed_clients = json.loads(key_data.get('allowedClients', '[]'))
    except:
        pass
    
    # Get model stats - daily from preloaded data
    model_stats_daily = []
    for redis_key, redis_value in all_redis_data.items():
        if redis_key.startswith(f'usage:{key_id}:model:daily:') and redis_key.endswith(f':{today}'):
            parts = redis_key.split(':')
            if len(parts) >= 6:
                model = parts[4]
                model_data = redis_value
                if model_data:
                    input_tokens = int(model_data.get('inputTokens', 0))
                    output_tokens = int(model_data.get('outputTokens', 0))
                    cache_read_tokens = int(model_data.get('cacheReadTokens', 0))
                    cache_create_tokens = int(model_data.get('cacheCreateTokens', 0))
                    
                    # Calculate cost based on model pricing
                    cost = 0.0
                    if model in MODEL_PRICING:
                        pricing = MODEL_PRICING[model]
                        input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                        output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                        cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                        cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                        cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                    
                    model_stats_daily.append({
                        'model': model,
                        'requests': int(model_data.get('requests', 0)),
                        'inputTokens': input_tokens,
                        'outputTokens': output_tokens,
                        'cacheReadTokens': cache_read_tokens,
                        'cacheCreateTokens': cache_create_tokens,
                        'allTokens': int(model_data.get('allTokens', 0)),
                        'cost': cost
                    })
    
    # Get model stats - monthly from preloaded data
    model_stats_monthly = []
    for redis_key, redis_value in all_redis_data.items():
        if redis_key.startswith(f'usage:{key_id}:model:monthly:') and redis_key.endswith(f':{current_month}'):
            parts = redis_key.split(':')
            if len(parts) >= 6:
                model = parts[4]
                model_data = redis_value
                if model_data:
                    input_tokens = int(model_data.get('inputTokens', 0))
                    output_tokens = int(model_data.get('outputTokens', 0))
                    cache_read_tokens = int(model_data.get('cacheReadTokens', 0))
                    cache_create_tokens = int(model_data.get('cacheCreateTokens', 0))
                    
                    # Calculate cost based on model pricing
                    cost = 0.0
                    if model in MODEL_PRICING:
                        pricing = MODEL_PRICING[model]
                        input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                        output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                        cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                        cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                        cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                    
                    model_stats_monthly.append({
                        'model': model,
                        'requests': int(model_data.get('requests', 0)),
                        'inputTokens': input_tokens,
                        'outputTokens': output_tokens,
                        'cacheReadTokens': cache_read_tokens,
                        'cacheCreateTokens': cache_create_tokens,
                        'allTokens': int(model_data.get('allTokens', 0)),
                        'cost': cost
                    })
    
    # Calculate window info similar to original function (simplified for preloaded data)
    window_current_requests = 0
    window_current_tokens = 0
    window_info = None
    
    # Get current window usage and session window info from preloaded data
    rate_limit_window = int(key_data.get('rateLimitWindow', 0))
    claude_account_id = key_data.get('claudeAccountId', '')
    
    if claude_account_id:
        account_key = f'claude:account:{claude_account_id}'
        if account_key in all_redis_data:
            account_data = all_redis_data[account_key]
            session_window_start = account_data.get('sessionWindowStart')
            session_window_end = account_data.get('sessionWindowEnd')
            
            if session_window_start and session_window_end:
                try:
                    window_start = datetime.fromisoformat(session_window_start.replace('Z', '+00:00'))
                    window_end = datetime.fromisoformat(session_window_end.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    
                    if window_start <= now <= window_end:
                        window_cost = calculate_window_cost_for_key_from_data(key_id, window_start, window_end, all_redis_data)
                        
                        # Convert to UTC+8 for display
                        window_start_tz8 = window_start + timedelta(hours=8)
                        window_end_tz8 = window_end + timedelta(hours=8)
                        
                        window_info = {
                            'start_hour': window_start_tz8.hour,
                            'start_min': window_start_tz8.minute,
                            'end_hour': window_end_tz8.hour,
                            'end_min': window_end_tz8.minute,
                            'cost': window_cost
                        }
                except Exception as e:
                    print(f"Error calculating window cost for key {key_id}: {e}")
    
    # Build response similar to original function
    result = {
        'name': key_data.get('name', 'Unknown'),
        'today_requests': int(daily_stats.get('requests', 0)),
        'today_tokens': int(daily_stats.get('allTokens', 0) or daily_stats.get('tokens', 0)),
        'today_cost': daily_cost,
        'today_input_tokens': int(daily_stats.get('inputTokens', 0)),
        'month_requests': int(monthly_stats.get('requests', 0)),
        'month_tokens': int(monthly_stats.get('allTokens', 0) or monthly_stats.get('tokens', 0)),
        'month_cost': monthly_cost,
        'month_input_tokens': int(monthly_stats.get('inputTokens', 0)),
        'daily_cost_limit': float(key_data.get('dailyCostLimit', 0)),
        'token_limit': int(key_data.get('tokenLimit', 0)),
        'rate_limit_window': int(key_data.get('rateLimitWindow', 0)),
        'rate_limit_requests': int(key_data.get('rateLimitRequests', 0)),
        'concurrency_limit': int(key_data.get('concurrencyLimit', 0)),
        'window_current_requests': window_current_requests,
        'window_current_tokens': window_current_tokens,
        'window_info': window_info,
        'model_stats_daily': model_stats_daily,
        'model_stats_monthly': model_stats_monthly,
        'permissions': key_data.get('permissions', 'unknown'),
        'is_active': key_data.get('isActive') == 'true',
        'last_used_at': format_last_used_at(key_data.get('lastUsedAt', ''), key_data.get('name', 'Unknown'), requesting_user),
        '_raw_last_used_at': key_data.get('lastUsedAt', ''),  # Add raw timestamp for sorting
        'status': None  # Will be calculated later when accounts info is available
    }
    
    # Include sensitive data only if requested
    if include_sensitive:
        result['id'] = key_id
        result['allowed_clients'] = allowed_clients
        result['created_at'] = key_data.get('createdAt', '')
    
    return result

def calculate_window_cost_for_key_from_data(key_id: str, window_start: datetime, window_end: datetime, all_redis_data: Dict[str, Any]) -> float:
    """Calculate window cost for a specific API key from preloaded data"""
    if not MODEL_PRICING:
        return 0.0
    
    total_cost = 0.0
    now = datetime.now(timezone.utc)
    
    # Make sure we don't go beyond current time
    if window_end > now:
        window_end = now
    
    # Calculate hours in the window
    hours_in_window = int((window_end - window_start).total_seconds() / 3600) + 1
    
    # Iterate through each hour in the window
    for hour_offset in range(hours_in_window):
        hour_time = window_start + timedelta(hours=hour_offset)
        if hour_time > window_end:
            break
        
        # Convert to UTC+8 for Redis key format
        hour_time_tz8 = hour_time + timedelta(hours=8)
        hour_str = hour_time_tz8.strftime('%Y-%m-%d')
        hour_num = hour_time_tz8.hour
        
        # Find all model usage keys for this hour and API key in preloaded data
        for redis_key, redis_value in all_redis_data.items():
            if (redis_key.startswith(f"usage:{key_id}:model:hourly:") and 
                redis_key.endswith(f":{hour_str}:{hour_num:02d}")):
                
                parts = redis_key.split(':')
                if len(parts) >= 7:
                    model_name = parts[4]
                    
                    # Get usage data
                    usage_data = redis_value
                    if usage_data and model_name in MODEL_PRICING:
                        pricing = MODEL_PRICING[model_name]
                        
                        input_tokens = int(usage_data.get('inputTokens', 0))
                        output_tokens = int(usage_data.get('outputTokens', 0))
                        cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                        cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                        
                        # Calculate cost
                        input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                        output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                        cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                        cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                        
                        hour_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                        total_cost += hour_cost
    
    return total_cost

def calculate_window_cost_from_data(claude_account_id: str, window_start: datetime, window_end: datetime, all_redis_data: Dict[str, Any]) -> float:
    """Calculate total cost for all models used within a session window from preloaded data"""
    if not MODEL_PRICING:
        return 0.0
    
    total_cost = 0.0
    now = datetime.now(timezone.utc)
    
    # Make sure we don't go beyond current time
    if window_end > now:
        window_end = now
    
    # Calculate hours in the window
    hours_in_window = int((window_end - window_start).total_seconds() / 3600) + 1
    
    # Iterate through each hour in the window
    for hour_offset in range(hours_in_window):
        hour_time = window_start + timedelta(hours=hour_offset)
        if hour_time > window_end:
            break
        
        # Convert to UTC+8 for Redis key format
        hour_time_tz8 = hour_time + timedelta(hours=8)
        hour_str = hour_time_tz8.strftime('%Y-%m-%d')
        hour_num = hour_time_tz8.hour
        
        # Find all model usage keys for this hour in preloaded data
        for redis_key, redis_value in all_redis_data.items():
            if (redis_key.startswith(f"account_usage:model:hourly:{claude_account_id}:") and
                redis_key.endswith(f":{hour_str}:{hour_num:02d}")):
                
                parts = redis_key.split(':')
                if len(parts) >= 7:
                    model_name = parts[4]
                    
                    # Get usage data
                    usage_data = redis_value
                    if usage_data and model_name in MODEL_PRICING:
                        pricing = MODEL_PRICING[model_name]
                        
                        input_tokens = int(usage_data.get('inputTokens', 0))
                        output_tokens = int(usage_data.get('outputTokens', 0))
                        cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                        cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                        
                        # Calculate cost including cache tokens
                        input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                        output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                        cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                        cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                        
                        hour_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                        total_cost += hour_cost
    
    return total_cost

def get_api_key_status(last_used_at: str, accounts: list) -> str:
    """Calculate API key status based on last usage time and window period
    
    Returns:
        'active' - Used within last 30 minutes
        'window' - Used outside 30 minutes but within window period
        'ok' - All other normal states
    """
    if not last_used_at or last_used_at == '':
        return 'ok'
    
    try:
        last_used_time = datetime.fromisoformat(last_used_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        
        # Check if there's a valid window
        window_start = None
        window_end = None
        
        if accounts and len(accounts) > 0:
            for account in accounts:
                if 'window_start' in account and 'window_end' in account:
                    ws = datetime.fromisoformat(account['window_start'].replace('Z', '+00:00'))
                    we = datetime.fromisoformat(account['window_end'].replace('Z', '+00:00'))
                    if now >= ws and now < we:
                        window_start = ws
                        window_end = we
                        break
        
        # Check if last used is 24 hours ago
        hours_diff = (now - last_used_time).total_seconds() / 3600
        if hours_diff >= 24:
            return 'ok'
        
        # Check if used within last 30 minutes
        minutes_diff = (now - last_used_time).total_seconds() / 60
        if minutes_diff <= 30:
            return 'active'
        
        # If there's a window period
        if window_start and window_end:
            # Check if last used is within window
            if last_used_time >= window_start and last_used_time < window_end:
                return 'window'
        
        # All other cases
        return 'ok'
    except:
        return 'ok'

def validate_key_name(name: str) -> bool:
    """Check if a key name exists in Redis"""
    all_keys = redis_client.keys('apikey:*')
    for key in all_keys:
        if key == 'apikey:hash_map':
            continue
        key_data = redis_client.hgetall(key)
        if key_data and key_data.get('name') == name and key_data.get('isActive') == 'true':
            return True
    return False

def user_exists(user_name: str) -> bool:
    """Check if a user exists in Redis by checking API keys"""
    try:
        all_keys = redis_client.keys('apikey:*')
        for key in all_keys:
            if key == 'apikey:hash_map':
                continue
            key_data = redis_client.hgetall(key)
            if key_data and key_data.get('name') == user_name and key_data.get('isActive') == 'true':
                return True
        return False
    except Exception as e:
        print(f"⚠️ Failed to check user existence: {e}")
        return False


def generate_all_detailed_stats(all_redis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate all detailed statistics from preloaded Redis data"""
    
    # Extract API keys and their data
    api_keys = {}
    hourly_usage_data = {}
    
    # Parse all Redis data
    for redis_key, redis_value in all_redis_data.items():
        if redis_key.startswith('apikey:') and redis_key != 'apikey:hash_map':
            key_id = redis_key.replace('apikey:', '')
            if redis_value and redis_value.get('isActive') == 'true':
                api_keys[key_id] = redis_value
        elif redis_key.startswith('usage:') and ':model:hourly:' in redis_key:
            # Parse hourly usage keys: usage:{key_id}:model:hourly:{model}:{date}:{hour}
            hourly_usage_data[redis_key] = redis_value
    
    # Initialize result structures
    api_detailed_stats = {}
    model_detailed_stats = {}
    all_models = set()
    
    # Get last 7 days of data
    today = get_date_in_timezone()
    date_range = [(today - timedelta(days=days_back)).strftime('%Y-%m-%d') for days_back in range(7)]
    
    # Initialize detailed stats structures
    for key_id, key_data in api_keys.items():
        api_name = key_data.get('name', f'API-{key_id}')
        api_detailed_stats[api_name] = {date: [] for date in date_range}
    
    # Process all hourly usage data
    for usage_key, usage_data in hourly_usage_data.items():
        if not usage_data:
            continue
            
        # Parse the usage key: usage:{key_id}:model:hourly:{model}:{date}:{hour}
        parts = usage_key.split(':')
        if len(parts) >= 7:
            key_id = parts[1]
            model = parts[4]
            date = parts[5]
            hour = int(parts[6])
            
            # Skip if date is not in our range
            if date not in date_range:
                continue
                
            # Skip if key is not active
            if key_id not in api_keys:
                continue
                
            # Add to all models set
            all_models.add(model)
            
            # Get API key name
            api_name = api_keys[key_id].get('name', f'API-{key_id}')
            
            # Calculate cost if we have pricing data
            if model in MODEL_PRICING:
                pricing = MODEL_PRICING[model]
                
                input_tokens = int(usage_data.get('inputTokens', 0))
                output_tokens = int(usage_data.get('outputTokens', 0))
                cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                
                input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                
                hourly_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                
                if hourly_cost > 0:
                    hour_minute = f"{hour:02d}:00"
                    
                    # Add to API detailed stats
                    if api_name not in api_detailed_stats:
                        api_detailed_stats[api_name] = {d: [] for d in date_range}
                    
                    # Find or create entry for this hour
                    day_entries = api_detailed_stats[api_name][date]
                    hour_entry = None
                    for entry in day_entries:
                        if entry['time'] == hour_minute:
                            hour_entry = entry
                            break
                    
                    if hour_entry is None:
                        hour_entry = {
                            'time': hour_minute,
                            'cost': 0.0,
                            'models': {}
                        }
                        day_entries.append(hour_entry)
                    
                    hour_entry['cost'] += hourly_cost
                    hour_entry['cost'] = round(hour_entry['cost'], 6)
                    if model not in hour_entry['models']:
                        hour_entry['models'][model] = 0.0
                    hour_entry['models'][model] += hourly_cost
                    
                    # Add to model detailed stats
                    if model not in model_detailed_stats:
                        model_detailed_stats[model] = {d: [] for d in date_range}
                    
                    # Find or create entry for this hour in model stats
                    model_day_entries = model_detailed_stats[model][date]
                    model_hour_entry = None
                    for entry in model_day_entries:
                        if entry['time'] == hour_minute:
                            model_hour_entry = entry
                            break
                    
                    if model_hour_entry is None:
                        model_hour_entry = {
                            'time': hour_minute,
                            'cost': 0.0,
                            'apis': {}
                        }
                        model_day_entries.append(model_hour_entry)
                    
                    model_hour_entry['cost'] += hourly_cost
                    model_hour_entry['cost'] = round(model_hour_entry['cost'], 6)
                    if api_name not in model_hour_entry['apis']:
                        model_hour_entry['apis'][api_name] = 0.0
                    model_hour_entry['apis'][api_name] += hourly_cost
    
    # Sort entries by time and remove empty days
    for api_name in api_detailed_stats:
        for date in list(api_detailed_stats[api_name].keys()):
            day_entries = api_detailed_stats[api_name][date]
            if day_entries:
                # Sort by time
                day_entries.sort(key=lambda x: x['time'])
            else:
                # Remove empty days
                del api_detailed_stats[api_name][date]
    
    for model in model_detailed_stats:
        for date in list(model_detailed_stats[model].keys()):
            day_entries = model_detailed_stats[model][date]
            if day_entries:
                # Sort by time
                day_entries.sort(key=lambda x: x['time'])
            else:
                # Remove empty days
                del model_detailed_stats[model][date]
    
    return {
        'api_detailed_stats': api_detailed_stats,
        'model_detailed_stats': model_detailed_stats
    }

@app.post("/api/stats/hourly")
async def get_hourly_detailed_stats(request: QueryRequest, http_request: Request):
    """Get detailed hourly cost statistics - requires API key or auth token"""
    
    # Check if auth_token is provided and valid
    if request.auth_token:
        if validate_auth_token(request.auth_token, http_request):
            print(f"✅ Authenticated via auth_token")
        else:
            raise HTTPException(status_code=401, detail="authfailed")
    # Check if API key is provided
    elif request.api_key:
        # Validate API key
        if not request.api_key.startswith('cr_'):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        # Check if API key exists
        key_id = validate_api_key(request.api_key)
        if not key_id:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Get user name from Redis
        key_data = redis_client.hgetall(f'apikey:{key_id}')
        user_name = key_data.get('name', 'Unknown') if key_data else 'Unknown'
        
        # Generate and store auth token
        auth_token = generate_auth_token(request.api_key)
        add_auth_token(auth_token, http_request, key_id, user_name)
        
        print(f"✅ Authenticated via API key: {key_id} ({user_name})")
    else:
        raise HTTPException(status_code=401, detail="API key or auth token required")
    
    # Get all Redis keys for model usage data
    print("🔍 Fetching hourly usage data...")
    all_redis_keys = redis_client.keys('*')
    
    # Filter for hourly usage keys and API keys
    hourly_keys = []
    api_keys = {}
    
    for key in all_redis_keys:
        if 'usage:' in key and ':model:hourly:' in key:
            hourly_keys.append(key)
        elif key.startswith('apikey:') and key != 'apikey:hash_map':
            key_data = redis_client.hgetall(key)
            if key_data and key_data.get('isActive') == 'true':
                key_id = key.replace('apikey:', '')
                api_keys[key_id] = key_data
    
    # Get last 7 days for detailed stats
    today = get_date_in_timezone()
    date_range = []
    for days_back in range(7):
        date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        date_range.append(date)
    
    # Process hourly data to create detailed breakdown
    detailed_stats = {}
    
    # Process each hourly usage key
    for usage_key in hourly_keys:
        # Parse: usage:{key_id}:model:hourly:{model}:{date}:{hour}
        parts = usage_key.split(':')
        if len(parts) >= 7:
            key_id = parts[1]
            model = parts[4]
            date = parts[5]
            hour = int(parts[6])
            
            # Skip if date not in our range or key not active
            if date not in date_range or key_id not in api_keys:
                continue
            
            # Get usage data
            usage_data = redis_client.hgetall(usage_key)
            if not usage_data:
                continue
                
            # Calculate cost if we have pricing data
            if model in MODEL_PRICING:
                pricing = MODEL_PRICING[model]
                
                input_tokens = int(usage_data.get('inputTokens', 0))
                output_tokens = int(usage_data.get('outputTokens', 0))
                cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                
                input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                
                hourly_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                
                if hourly_cost > 0:
                    # Initialize date structure if not exists
                    if date not in detailed_stats:
                        detailed_stats[date] = {}
                    
                    # Create time slot key (00:00 - 00:01 format)
                    start_time = f"{hour:02d}:00"
                    end_time = f"{hour:02d}:59" if hour < 23 else "23:59"
                    time_slot = f"{start_time} - {end_time}"
                    
                    if time_slot not in detailed_stats[date]:
                        detailed_stats[date][time_slot] = 0.0
                    
                    detailed_stats[date][time_slot] += hourly_cost
    
    # Sort dates and format response
    sorted_dates = sorted(detailed_stats.keys(), reverse=True)
    formatted_stats = {}
    
    for date in sorted_dates:
        # Sort time slots within each date
        time_slots = detailed_stats[date]
        sorted_time_slots = sorted(time_slots.items(), key=lambda x: x[0])
        
        formatted_stats[date] = []
        for time_slot, cost in sorted_time_slots:
            formatted_stats[date].append({
                'time': time_slot,
                'cost': cost
            })
    
    return {
        'success': True,
        'detailed_stats': formatted_stats,
        'query_time': datetime.now(timezone.utc).isoformat()
    }

@app.post("/simple-board/query_all")
async def get_all_stats(request: QueryRequest, http_request: Request, leave_message: Optional[str] = None):
    """Get statistics for all API keys with detailed breakdown - requires API key or auth token"""
    
    # Variables for logging
    auth_method = ""
    masked_key = ""
    auth_token_for_log = ""
    user_id_for_log = ""
    requesting_user_name = ""  # Track the requesting user's name for SHOW_EXACT_TIME_LIST check
    
    # Check if auth_token is provided and valid
    if request.auth_token:
        if validate_auth_token(request.auth_token, http_request):
            print(f"✅ Authenticated via auth_token")
            auth_method = "auth_token"
            masked_key = "-"
            auth_token_for_log = request.auth_token
            # Get user_id and user_name from token cache
            token_info = auth_token_cache.get(request.auth_token, {})
            user_id_for_log = token_info.get('user_id', 'unknown')
            requesting_user_name = token_info.get('user_name', '')
        else:
            raise HTTPException(status_code=401, detail="authfailed")
    # Check if API key is provided
    elif request.api_key:
        # Validate API key
        if not request.api_key.startswith('cr_'):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        # Check if API key exists
        key_id = validate_api_key(request.api_key)
        if not key_id:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Get user name from Redis
        key_data = redis_client.hgetall(f'apikey:{key_id}')
        user_name = key_data.get('name', 'Unknown') if key_data else 'Unknown'
        
        # Generate and store auth token
        auth_token = generate_auth_token(request.api_key)
        add_auth_token(auth_token, http_request, key_id, user_name)
        
        print(f"✅ Authenticated via API key: {key_id} ({user_name})")
        auth_method = "api_key"
        masked_key = mask_api_key(request.api_key)
        auth_token_for_log = auth_token
        user_id_for_log = key_id
        requesting_user_name = user_name  # Store for SHOW_EXACT_TIME_LIST check
    else:
        raise HTTPException(status_code=401, detail="API key or auth token required")
    
    # Handle leave_message parameter if provided and user is authenticated
    if leave_message is not None and requesting_user_name:
        # First check if the user exists in Redis
        if not user_exists(requesting_user_name):
            print(f"⚠️ User {requesting_user_name} does not exist in Redis, ignoring message")
        else:
            try:
                if leave_message == "":
                    # Empty message means clear the message
                    if requesting_user_name in user_messages_cache:
                        del user_messages_cache[requesting_user_name]
                        print(f"✉️ Cleared message for user: {requesting_user_name}")
                else:
                    # Limit message to 300 characters
                    truncated_message = leave_message[:300] if len(leave_message) > 300 else leave_message
                    if len(leave_message) > 300:
                        print(f"✂️ Truncated message from {len(leave_message)} to 300 characters")
                    
                    # Store the message in memory cache
                    user_messages_cache[requesting_user_name] = {
                        'message': truncated_message,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    print(f"✉️ Stored message for user: {requesting_user_name} (length: {len(truncated_message)})")
            except Exception as e:
                print(f"⚠️ Failed to store message: {e}")
    
    # Log the request
    client_ip = get_client_ip(http_request)
    user_agent = http_request.headers.get('User-Agent', 'Unknown')
    log_request("/simple-board/query_all", client_ip, user_agent, auth_method, 
               masked_key, auth_token_for_log, user_id_for_log)
    
    # 🚀 Optimized Redis data fetch - only get keys we need
    print("🔍 Fetching required Redis keys...")
    
    # Only fetch keys we actually use for statistics
    required_key_patterns = [
        'usage:*',          # All usage-related keys (daily, monthly, model, cost)
        'apikey:*',         # API key configurations 
        'claude:account:*', # Claude account data for session windows
        'account_usage:*'   # Account usage data
    ]
    
    all_redis_keys = []
    for pattern in required_key_patterns:
        pattern_keys = redis_client.keys(pattern)
        all_redis_keys.extend(pattern_keys)
    
    print(f"📊 Found {len(all_redis_keys)} required Redis keys")
    
    # Batch fetch all Redis data in one go
    print("📦 Batch loading required Redis data...")
    all_redis_data = {}
    
    # Categorize keys by type and expected Redis command
    cost_keys = []
    hash_keys = []
    
    for key in all_redis_keys:
        if key.startswith('usage:cost:'):
            cost_keys.append(key)
        elif (key.startswith('apikey:') or 
              key.startswith('usage:daily:') or 
              key.startswith('usage:monthly:') or 
              key.startswith('usage:') and ':model:' in key or
              key.startswith('claude:account:') or
              key.startswith('account_usage:')):
            hash_keys.append(key)
    
    # Use separate pipelines for different command types to avoid WRONGTYPE errors
    
    # Pipeline 1: GET commands for cost keys (string values)
    if cost_keys:
        pipeline1 = redis_client.pipeline()
        
        for key in cost_keys:
            pipeline1.get(key)
        
        try:
            string_results = pipeline1.execute()
            
            # Map cost key results
            for i, key in enumerate(cost_keys):
                result = string_results[i]
                if result:  # Only store non-empty results
                    all_redis_data[key] = result
                    
        except Exception as e:
            print(f"⚠️ Error executing cost keys pipeline: {e}")
    
    # Pipeline 2: HGETALL commands for hash keys
    if hash_keys:
        pipeline2 = redis_client.pipeline()
        
        for key in hash_keys:
            pipeline2.hgetall(key)
        
        try:
            hash_results = pipeline2.execute()
            
            # Map hash key results  
            for i, key in enumerate(hash_keys):
                result = hash_results[i]
                if result:  # Only store non-empty results
                    all_redis_data[key] = result
                    
        except Exception as e:
            print(f"⚠️ Error executing hash keys pipeline: {e}")
            # If hash pipeline fails, try individual keys to identify problematic ones
            print(f"🔍 Attempting individual hash key retrieval...")
            for key in hash_keys:
                try:
                    result = redis_client.hgetall(key)
                    if result:
                        all_redis_data[key] = result
                except Exception as key_error:
                    print(f"❌ Failed to get hash key '{key}': {key_error}")
                    # Try as string value instead
                    try:
                        string_result = redis_client.get(key)
                        if string_result:
                            all_redis_data[key] = string_result
                            print(f"✅ Successfully got '{key}' as string value")
                    except Exception as string_error:
                        print(f"❌ Failed to get '{key}' as string too: {string_error}")
    
    print(f"✅ Loaded {len(all_redis_data)} non-empty Redis entries")
    
    # Process API keys from loaded data
    api_stats = []
    all_keys = []
    
    for key in all_redis_keys:
        if key.startswith('apikey:') and key != 'apikey:hash_map':
            all_keys.append(key)
    
    print(f"🔑 Processing {len(all_keys)} API keys...")
    
    for key in all_keys:
        key_id = key.replace('apikey:', '')
        
        # Get stats for this key using preloaded data
        stats = get_api_key_stats_from_data(key_id, all_redis_data, include_sensitive=True, requesting_user=requesting_user_name)
        if stats:
            api_stats.append(stats)
            print(f"  ✅ Added stats for: {stats['name']} ({key_id})")
        else:
            print(f"  ⏭️  Skipped inactive key: {key_id}")
    
    # Sort by last used time and assign sort values
    api_stats_with_time = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    
    for stat in api_stats:
        # Parse the raw timestamp if available
        raw_timestamp = stat.get('_raw_last_used_at', '')
        if raw_timestamp:
            try:
                # Parse ISO format timestamp
                last_used_dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                api_stats_with_time.append((stat, last_used_dt, stat.get('name', '')))
            except:
                # If parsing fails, use epoch as fallback
                api_stats_with_time.append((stat, epoch, stat.get('name', '')))
        else:
            # No timestamp, use epoch
            api_stats_with_time.append((stat, epoch, stat.get('name', '')))
    
    # Sort by:
    # 1. Timestamp (newest first) - users with recent activity come first
    # 2. Name (alphabetically) - for users with no recent activity (epoch timestamp)
    api_stats_with_time.sort(key=lambda x: (-x[1].timestamp(), x[2]))
    
    # Assign sort values and rebuild api_stats (keep _raw_last_used_at for now)
    api_stats = []
    for idx, (stat, _, _) in enumerate(api_stats_with_time):
        # The most recently used gets the highest sort value
        stat['sort'] = len(api_stats_with_time) - idx
        # Keep the raw timestamp for status calculation
        api_stats.append(stat)
    
    # Get all unique Claude accounts with session windows from loaded data
    accounts = []
    seen_account_ids = set()
    
    for stats in api_stats:
        claude_account_id = None
        # Get the claudeAccountId from the preloaded data
        api_key = f"apikey:{stats['id']}"
        if api_key in all_redis_data:
            claude_account_id = all_redis_data[api_key].get('claudeAccountId')
        
        if claude_account_id and claude_account_id not in seen_account_ids:
            account_key = f'claude:account:{claude_account_id}'
            if account_key in all_redis_data:
                account_data = all_redis_data[account_key]
                if account_data.get('sessionWindowStart') and account_data.get('sessionWindowEnd'):
                    window_start_str = account_data.get('sessionWindowStart')
                    window_end_str = account_data.get('sessionWindowEnd')
                    
                    # Calculate window cost using preloaded data
                    window_cost = 0.0
                    try:
                        window_start = datetime.fromisoformat(window_start_str.replace('Z', '+00:00'))
                        window_end = datetime.fromisoformat(window_end_str.replace('Z', '+00:00'))
                        window_cost = calculate_window_cost_from_data(claude_account_id, window_start, window_end, all_redis_data)
                    except Exception as e:
                        print(f"Error calculating window cost for account {claude_account_id}: {e}")
                    
                    accounts.append({
                        'name': account_data.get('name', 'Unknown'),
                        'window_start': window_start_str,
                        'window_end': window_end_str,
                        'window_cost': window_cost,
                        'window_limit': 120.0  # $120 limit per window
                    })
                    seen_account_ids.add(claude_account_id)
    
    # Now calculate status for each API key using the accounts information
    for stat in api_stats:
        raw_timestamp = stat.get('_raw_last_used_at', '')
        stat['status'] = get_api_key_status(raw_timestamp, accounts)
        # Remove the internal raw timestamp field from response
        stat.pop('_raw_last_used_at', None)
    
    # Generate ALL detailed stats from preloaded data in one go
    print("📈 Generating detailed statistics...")
    detailed_results = generate_all_detailed_stats(all_redis_data)
    detailed_stats = detailed_results['api_detailed_stats']
    model_detailed_stats = detailed_results['model_detailed_stats']
    
    # Get user messages from memory cache and validate users still exist
    validated_messages = {}
    for user_name, msg_data in user_messages_cache.items():
        # Check if user still exists in Redis
        if user_exists(user_name):
            validated_messages[user_name] = msg_data
        else:
            print(f"⚠️ User {user_name} no longer exists, skipping message")
    
    # Clean up sensitive data from api_stats before returning and add leave_message field
    cleaned_stats = []
    for stat in api_stats:
        # Create a copy without sensitive fields
        cleaned_stat = {k: v for k, v in stat.items() if k not in ['id', 'allowed_clients', 'created_at']}
        # Add leave_message field if this user has a message
        user_name = stat.get('name', '')
        if user_name in validated_messages:
            cleaned_stat['leave_message'] = validated_messages[user_name]
        else:
            cleaned_stat['leave_message'] = None
        cleaned_stats.append(cleaned_stat)
    
    response_data = {
        'success': True,
        'count': len(cleaned_stats),
        'data': cleaned_stats,
        'accounts': accounts,
        'detailed_stats': detailed_stats,
        'model_detailed_stats': model_detailed_stats,
        'model_map': MODEL_MAP,  # Add model map to response
        'query_time': datetime.now(timezone.utc).isoformat(),
        'authenticated_user': requesting_user_name  # Add authenticated user name
    }
    
    # If authenticated via API key, include the auth token in response
    if request.api_key and not request.auth_token:
        response_data['auth_token'] = generate_auth_token(request.api_key)
    
    return response_data


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "redis": "disconnected", "error": str(e)}
        )


@app.get("/index.html")
async def get_board_index():
    """Serve the dashboard HTML file"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard HTML file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading HTML file: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "API Stats Query Service V2",
        "version": "2.1",
        "endpoints": {
            "GET /simple-board/index.html": "Dashboard web interface",
            "POST /simple-board/query_all": "Get all API keys statistics with detailed breakdown (requires authentication)",
            "GET /health": "Health check",
            "GET /": "This help message"
        },
        "authentication": "Use 'api_key' (cr_ prefix) or 'auth_token' for authentication",
        "features": [
            "Persistent auth token storage",
            "Detailed hourly statistics for each API key",
            "Model-specific usage breakdown",
            "IP address tracking and session management"
        ]
    }

def generate_model_map(models):
    """Generate a display name mapping for models"""
    model_map = {}
    
    for model_name in models:
        # Skip sample spec and non-model entries
        if model_name == 'sample_spec':
            continue
            
        # Format display name based on patterns
        display_name = model_name
        
        # Claude models
        if 'claude' in model_name:
            if 'haiku' in model_name:
                if '3-5' in model_name or '3.5' in model_name:
                    display_name = 'Haiku-3.5'
                else:
                    display_name = 'Haiku-3'
            elif 'sonnet' in model_name:
                if 'sonnet-4-20250514' in model_name:
                    display_name = 'Sonnet-4.1'
                elif 'sonnet-4' in model_name or 'claude-4' in model_name:
                    display_name = 'Sonnet-4'
                elif '3-7' in model_name or '3.7' in model_name:
                    display_name = 'Sonnet-3.7'
                elif '3-5' in model_name or '3.5' in model_name:
                    display_name = 'Sonnet-3.5'
                else:
                    display_name = 'Sonnet-3'
            elif 'opus' in model_name:
                if 'opus-4-1' in model_name or 'opus-4.1' in model_name:
                    display_name = 'Opus-4.1'
                elif 'opus-4' in model_name or 'claude-4' in model_name:
                    display_name = 'Opus-4'
                else:
                    display_name = 'Opus-3'
        
        # GPT models
        elif 'gpt' in model_name:
            if 'gpt-4.1' in model_name:
                if 'mini' in model_name:
                    display_name = 'GPT-4.1-Mini'
                elif 'nano' in model_name:
                    display_name = 'GPT-4.1-Nano'
                else:
                    display_name = 'GPT-4.1'
            elif 'gpt-4.5' in model_name:
                display_name = 'GPT-4.5'
            elif 'gpt-5' in model_name:
                if 'mini' in model_name:
                    display_name = 'GPT-5-Mini'
                elif 'nano' in model_name:
                    display_name = 'GPT-5-Nano'
                else:
                    display_name = 'GPT-5'
            elif 'gpt-4o' in model_name:
                if 'mini' in model_name:
                    display_name = 'GPT-4o-Mini'
                elif 'realtime' in model_name:
                    display_name = 'GPT-4o-Realtime'
                elif 'audio' in model_name:
                    display_name = 'GPT-4o-Audio'
                else:
                    display_name = 'GPT-4o'
            elif 'gpt-4-turbo' in model_name:
                display_name = 'GPT-4-Turbo'
            elif 'gpt-4' in model_name:
                if '32k' in model_name:
                    display_name = 'GPT-4-32K'
                else:
                    display_name = 'GPT-4'
            elif 'gpt-3.5-turbo' in model_name:
                if '16k' in model_name:
                    display_name = 'GPT-3.5-16K'
                else:
                    display_name = 'GPT-3.5'
        
        # O1/O3/O4 models
        elif model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
            parts = model_name.split('-')
            if len(parts) > 0:
                base = parts[0].upper()
                if 'mini' in model_name:
                    display_name = f'{base}-Mini'
                elif 'pro' in model_name:
                    display_name = f'{base}-Pro'
                elif 'deep-research' in model_name:
                    display_name = f'{base}-Deep-Research'
                else:
                    display_name = base
        
        # DeepSeek models
        elif 'deepseek' in model_name:
            if 'coder' in model_name:
                display_name = 'DeepSeek-Coder'
            elif 'reasoner' in model_name:
                display_name = 'DeepSeek-Reasoner'
            elif 'r1' in model_name:
                display_name = 'DeepSeek-R1'
            elif 'v3' in model_name:
                display_name = 'DeepSeek-V3'
            else:
                display_name = 'DeepSeek'
        
        # Mixtral models
        elif 'mixtral' in model_name:
            if '8x7b' in model_name.lower():
                display_name = 'Mixtral-8x7B'
            elif '8x22b' in model_name.lower():
                display_name = 'Mixtral-8x22B'
            else:
                display_name = 'Mixtral'
        
        # Llama models
        elif 'llama' in model_name:
            if 'llama-3' in model_name or 'llama3' in model_name:
                if '70b' in model_name:
                    display_name = 'Llama3-70B'
                elif '8b' in model_name:
                    display_name = 'Llama3-8B'
                else:
                    display_name = 'Llama3'
            elif 'llama-2' in model_name or 'llama2' in model_name:
                if '70b' in model_name:
                    display_name = 'Llama2-70B'
                elif '13b' in model_name:
                    display_name = 'Llama2-13B'
                elif '7b' in model_name:
                    display_name = 'Llama2-7B'
                else:
                    display_name = 'Llama2'
            else:
                display_name = 'Llama'
        
        # Gemini models
        elif 'gemini' in model_name:
            if 'pro' in model_name:
                display_name = 'Gemini-Pro'
            elif 'flash' in model_name:
                display_name = 'Gemini-Flash'
            else:
                display_name = 'Gemini'
        
        # Add to map if we generated a reasonable display name
        if display_name != model_name or '/' not in model_name:
            model_map[model_name] = display_name
    
    return model_map

def load_model_pricing():
    """Load model pricing from JSON file and generate model map"""
    global MODEL_PRICING, MODEL_MAP
    try:
        pricing_file = os.path.join(os.path.dirname(__file__), 'model_pricing.json')
        with open(pricing_file, 'r') as f:
            MODEL_PRICING = json.load(f)
            print(f"✅ Loaded {len(MODEL_PRICING)} model pricing entries")
            
            # Generate model map from loaded models
            MODEL_MAP = generate_model_map(MODEL_PRICING.keys())
            print(f"✅ Generated model map with {len(MODEL_MAP)} entries")
    except Exception as e:
        print(f"⚠️ Failed to load model pricing: {e}")
        MODEL_PRICING = {}
        MODEL_MAP = {}

def calculate_window_cost_for_key(key_id: str, window_start: datetime, window_end: datetime) -> float:
    """Calculate window cost for a specific API key"""
    if not MODEL_PRICING:
        return 0.0
    
    total_cost = 0.0
    now = datetime.now(timezone.utc)
    
    # Make sure we don't go beyond current time
    if window_end > now:
        window_end = now
    
    # Calculate hours in the window
    hours_in_window = int((window_end - window_start).total_seconds() / 3600) + 1
    
    # Iterate through each hour in the window
    for hour_offset in range(hours_in_window):
        hour_time = window_start + timedelta(hours=hour_offset)
        if hour_time > window_end:
            break
        
        # Convert to UTC+8 for Redis key format
        hour_time_tz8 = hour_time + timedelta(hours=8)
        hour_str = hour_time_tz8.strftime('%Y-%m-%d')
        hour_num = hour_time_tz8.hour
        
        # Find all model usage keys for this hour and API key
        pattern = f"usage:{key_id}:model:hourly:*:{hour_str}:{hour_num:02d}"
        model_keys = redis_client.keys(pattern)
        
        for key in model_keys:
            parts = key.split(':')
            if len(parts) >= 7:
                model_name = parts[4]
                
                # Get usage data
                usage_data = redis_client.hgetall(key)
                if usage_data and model_name in MODEL_PRICING:
                    pricing = MODEL_PRICING[model_name]
                    
                    input_tokens = int(usage_data.get('inputTokens', 0))
                    output_tokens = int(usage_data.get('outputTokens', 0))
                    cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                    cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                    
                    # Calculate cost
                    input_cost = input_tokens * pricing.get('input_cost_per_token', 0)
                    output_cost = output_tokens * pricing.get('output_cost_per_token', 0)
                    cache_read_cost = cache_read_tokens * pricing.get('cache_read_input_token_cost', 0)
                    cache_create_cost = cache_create_tokens * pricing.get('cache_creation_input_token_cost', 0)
                    
                    hour_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                    total_cost += hour_cost
    
    return total_cost

def calculate_window_cost(claude_account_id: str, window_start: datetime, window_end: datetime) -> float:
    """Calculate total cost for all models used within a session window"""
    print(f"\n📊 Calculating window cost for account: {claude_account_id}")
    
    # Convert window times to UTC+8 for display
    window_start_tz8 = window_start + timedelta(hours=8)
    window_end_tz8 = window_end + timedelta(hours=8)
    
    print(f"   Window start: {window_start_tz8.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
    print(f"   Window end: {window_end_tz8.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
    
    if not MODEL_PRICING:
        print(f"   ❌ No model pricing data loaded")
        return 0.0
    
    total_cost = 0.0
    now = datetime.now(timezone.utc)
    now_tz8 = now + timedelta(hours=8)
    print(f"   Current time: {now_tz8.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
    
    # Make sure we don't go beyond current time
    if window_end > now:
        print(f"   ⚠️ Window end is in future, adjusting to current time")
        window_end = now
    
    # Calculate hours in the window
    hours_in_window = int((window_end - window_start).total_seconds() / 3600) + 1
    print(f"   Hours in window: {hours_in_window}")
    
    # Iterate through each hour in the window
    hours_with_data = 0
    for hour_offset in range(hours_in_window):
        hour_time = window_start + timedelta(hours=hour_offset)
        if hour_time > window_end:
            break
        
        # Convert to UTC+8 for Redis key format (since Redis keys are stored in UTC+8)
        hour_time_tz8 = hour_time + timedelta(hours=8)
        
        # Get all model usage keys for this account and hour
        hour_str = hour_time_tz8.strftime('%Y-%m-%d')
        hour_num = hour_time_tz8.hour
        
        # Find all model usage keys for this hour
        pattern = f"account_usage:model:hourly:{claude_account_id}:*:{hour_str}:{hour_num:02d}"
        model_keys = redis_client.keys(pattern)
        
        if model_keys:
            hours_with_data += 1
            print(f"   Hour {hour_offset}: {hour_str} {hour_num:02d}:00 (UTC+8) - Found {len(model_keys)} model keys")
        
        for key in model_keys:
            # Extract model name from key
            parts = key.split(':')
            if len(parts) >= 7:
                model_name = parts[4]
                
                # Get usage data
                usage_data = redis_client.hgetall(key)
                if usage_data:
                    input_tokens = int(usage_data.get('inputTokens', 0))
                    output_tokens = int(usage_data.get('outputTokens', 0))
                    cache_read_tokens = int(usage_data.get('cacheReadTokens', 0))
                    cache_create_tokens = int(usage_data.get('cacheCreateTokens', 0))
                    
                    # Find pricing for this model
                    if model_name in MODEL_PRICING:
                        pricing = MODEL_PRICING[model_name]
                        input_cost_per_token = pricing.get('input_cost_per_token', 0)
                        output_cost_per_token = pricing.get('output_cost_per_token', 0)
                        cache_read_cost_per_token = pricing.get('cache_read_input_token_cost', 0)
                        cache_create_cost_per_token = pricing.get('cache_creation_input_token_cost', 0)
                        
                        # Calculate cost including cache tokens
                        input_cost = input_tokens * input_cost_per_token
                        output_cost = output_tokens * output_cost_per_token
                        cache_read_cost = cache_read_tokens * cache_read_cost_per_token
                        cache_create_cost = cache_create_tokens * cache_create_cost_per_token
                        
                        hour_cost = input_cost + output_cost + cache_read_cost + cache_create_cost
                        total_cost += hour_cost
                        
                        print(f"      - Model: {model_name}")
                        if input_tokens > 0:
                            print(f"        Input: {input_tokens} tokens @ ${input_cost_per_token:.10f} = ${input_cost:.6f}")
                        if output_tokens > 0:
                            print(f"        Output: {output_tokens} tokens @ ${output_cost_per_token:.10f} = ${output_cost:.6f}")
                        if cache_read_tokens > 0:
                            print(f"        Cache Read: {cache_read_tokens} tokens @ ${cache_read_cost_per_token:.10f} = ${cache_read_cost:.6f}")
                        if cache_create_tokens > 0:
                            print(f"        Cache Create: {cache_create_tokens} tokens @ ${cache_create_cost_per_token:.10f} = ${cache_create_cost:.6f}")
                        print(f"        Hour cost: ${hour_cost:.6f}")
                    else:
                        print(f"      ⚠️ Model '{model_name}' not found in pricing data")
                        print(f"         Input: {input_tokens}, Output: {output_tokens}, Cache Read: {cache_read_tokens}, Cache Create: {cache_create_tokens}")
    
    print(f"   📈 Total hours with data: {hours_with_data}")
    print(f"   💰 Total window cost: ${total_cost:.6f}")
    return total_cost

def main():
    """Main function to run the FastAPI server"""
    print("=" * 80)
    print("Starting API Stats Query Service V2...")
    print("=" * 80)
    
    # Load model pricing on startup
    load_model_pricing()
    
    # Load auth tokens on startup
    load_auth_tokens()
    
    print("Server: http://127.0.0.1:8001")
    print("\nEndpoints:")
    print("  POST /simple-board/query_all - Get all API keys statistics with detailed breakdown (requires auth)")
    print("  GET  /health          - Health check")
    print("  GET  /                - API information")
    print("\nAuthentication:")
    print("  - Use 'api_key' field with a valid cr_ key")
    print("  - Or use 'auth_token' for authenticated requests")
    print("  - Auth tokens are persistent and include IP tracking")
    print("\nFeatures:")
    print("  - Detailed hourly statistics for each API key")
    print("  - Model-specific usage breakdown")
    print("  - Persistent auth token storage with IP tracking")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()