# System Design Interview Guide

## Overview
System design interviews test your ability to architect large-scale distributed systems. The key is to think systematically, ask clarifying questions, and make reasonable trade-offs while discussing your reasoning out loud.

## 1. Functional Requirements

### Approach
- **Start with use cases**: What does the system need to do?
- **Focus on core functionality first**: Don't get lost in edge cases initially
- **Backend-focused**: Prioritize API design and data flow over UI details

### Key Questions to Ask
- Who are the users and what are their primary actions?
- What are the core features vs nice-to-have features?
- What types of data will the system handle?
- Are there any special requirements (real-time, offline support, etc.)?

### Example Use Cases
**Data Labeling Platform:**
- Users upload datasets (video/text)
- Labelers download and annotate data
- System tracks labeling progress and quality
- Admins manage user permissions and dataset access

**URL Shortener (like bit.ly):**
- Users create short URLs from long URLs
- Users click short URLs and get redirected
- Analytics on click rates and usage patterns

## 2. Non-Functional Requirements

### Scale Estimation
Always estimate scale to drive architectural decisions.

#### Traffic Scale
- **Requests per second (RPS)**: Start with given numbers
- **Growth factor**: Plan for 10x growth
- **Peak vs average**: Peak can be 3-5x average

**Example Calculation:**
```
Given: 10,000 RPS average
Peak: 30,000 RPS
With growth: 100,000 RPS peak
Servers needed: 100,000 ÷ 10,000 = 10 servers (add 50% overhead = 15 servers)
```

#### Data Scale
**Text Data:**
- Average article: ~3 KB
- 1 million articles = ~3 GB
- Consider compression (can reduce by 50-70%)

**Video Data:**
- HD video: ~5 MB/minute
- 4K video: ~25 MB/minute
- Example: 41k videos × 60 MB avg = ~2.5 TB

**Images:**
- High resolution: ~5-10 MB
- Thumbnail: ~50-100 KB
- Profile picture: ~200 KB

#### Storage Growth
- Plan for 3-5 years of growth
- Consider data retention policies
- Factor in backups and replication

### Availability Requirements
- **99.9% (8.77 hours downtime/year)**: Acceptable for internal tools
- **99.99% (52.6 minutes downtime/year)**: Standard for customer-facing
- **99.999% (5.26 minutes downtime/year)**: Critical financial systems

### Latency Requirements
Focus on percentiles, not just averages:
- **P50 (median)**: 50% of requests
- **P95**: 95% of requests (more important for user experience)
- **P99**: 99% of requests (catches tail latency)

**Typical Targets:**
- API responses: P95 < 100ms
- Database queries: P95 < 10ms
- Page loads: P95 < 2 seconds
- Search results: P95 < 200ms

## 3. High-Level Design & Key APIs

### API Design Principles
- RESTful when possible
- Consistent naming conventions
- Proper HTTP status codes
- Version your APIs (/v1/, /v2/)

### Example API Design
```
Data Labeling Platform:

GET /v1/datasets
GET /v1/datasets/{dataset_id}
POST /v1/datasets/{dataset_id}/download
POST /v1/datasets/{dataset_id}/labels
PUT /v1/datasets/{dataset_id}/labels/{label_id}
GET /v1/users/{user_id}/assignments

Authentication: Bearer token in Authorization header
Rate limiting: 1000 requests/hour per user
```

### System Architecture Components

```
Client Apps → Load Balancer → API Gateway → Microservices → Databases
                     ↓
               CDN (for static content)
                     ↓
               Object Storage (S3)
```

## 4. Load Balancing

### Load Balancer Types
1. **Layer 4 (Transport)**: Routes based on IP and port
2. **Layer 7 (Application)**: Routes based on HTTP headers, URLs

### Load Balancing Algorithms
- **Round Robin**: Simple, even distribution
- **Least Connections**: Routes to server with fewest active connections
- **Weighted Round Robin**: Assign weights based on server capacity
- **IP Hash**: Route same IP to same server (session affinity)
- **Geographic**: Route based on user location

### High Availability
- Multiple load balancers with failover
- Health checks to remove unhealthy servers
- Auto-scaling based on metrics

## 5. Database Design

### SQL vs NoSQL Decision Matrix

| Use SQL When: | Use NoSQL When: |
|---------------|-----------------|
| Complex relationships and joins | Simple key-value or document storage |
| ACID transactions required | High write throughput needed |
| Strong consistency needed | Horizontal scaling required |
| Complex queries and reporting | Schema flexibility important |
| Well-defined, stable schema | Rapid development cycles |

### NoSQL Database Types

#### Key-Value Stores
- **Examples**: Redis, DynamoDB
- **Use cases**: Caching, session storage, simple lookups
- **Pros**: Extremely fast, simple
- **Cons**: Limited query capabilities

#### Document Stores
- **Examples**: MongoDB, CouchDB
- **Use cases**: Content management, catalogs, user profiles
- **Pros**: Flexible schema, JSON-like documents
- **Cons**: No joins, eventual consistency

#### Wide Column
- **Examples**: Cassandra, HBase
- **Use cases**: Time-series data, IoT data, logging
- **Pros**: High write throughput, good for time-series
- **Cons**: Limited query flexibility

#### Graph Databases
- **Examples**: Neo4j, Amazon Neptune
- **Use cases**: Social networks, recommendation engines
- **Pros**: Great for relationship queries
- **Cons**: Complex to scale, specialized use cases

### Scaling Strategies

#### Vertical Scaling (Scale Up)
- Add more CPU, RAM, storage to existing server
- **Pros**: Simple, no code changes
- **Cons**: Hardware limits, expensive, single point of failure

#### Horizontal Scaling (Scale Out)

**Federation (Functional Partitioning)**
- Split databases by feature (users, posts, messages)
- Each service owns its data
- **Pros**: Clear boundaries, team ownership
- **Cons**: Cross-service queries difficult

**Sharding (Horizontal Partitioning)**
- Split same data type across multiple databases
- **Pros**: Linear scaling, better performance
- **Cons**: Complex queries, rebalancing overhead

**Sharding Strategies:**
1. **Range-based**: Users A-M on server 1, N-Z on server 2
2. **Hash-based**: hash(user_id) % num_servers
3. **Directory-based**: Lookup service maintains shard mappings

## 6. Consistent Hashing

### Problem with Simple Hashing
- Adding/removing servers requires rehashing most data
- Example: hash(key) % 4 servers → hash(key) % 5 servers

### Consistent Hashing Solution
1. **Hash Ring**: Map servers and data to points on a circle
2. **Data Placement**: Each key goes to the next server clockwise
3. **Adding Servers**: Only affects data between new server and previous server
4. **Virtual Nodes**: Each physical server gets multiple points on ring for better distribution

### Benefits
- Minimal data movement when scaling
- Built into DynamoDB, Cassandra
- Handles hot spots better with virtual nodes

## 7. Detailed Architecture

### Database Schema Examples

#### SQL Schema (PostgreSQL)
```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Datasets table
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    size_bytes BIGINT,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Labels table
CREATE TABLE labels (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    labeller_id INTEGER REFERENCES users(id),
    data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### NoSQL Schema (DynamoDB)
```json
// Users table
{
  "TableName": "Users",
  "PartitionKey": "user_id",
  "Attributes": {
    "user_id": "string",
    "email": "string", 
    "role": "string",
    "created_at": "timestamp"
  }
}

// Labels table
{
  "TableName": "Labels",
  "PartitionKey": "dataset_id",
  "SortKey": "label_id",
  "GSI": "labeller_id-created_at-index",
  "Attributes": {
    "dataset_id": "string",
    "label_id": "string",
    "labeller_id": "string",
    "label_data": "map",
    "status": "string",
    "created_at": "timestamp"
  }
}
```

### Caching Strategy

#### Cache Levels
1. **Browser Cache**: Static assets, API responses
2. **CDN**: Global edge caches for static content
3. **Application Cache**: In-memory (Redis) for frequently accessed data
4. **Database Cache**: Query result caching

#### Cache Patterns
- **Cache-Aside**: App manages cache, reads from DB on miss
- **Write-Through**: Write to cache and DB simultaneously
- **Write-Behind**: Write to cache immediately, DB asynchronously
- **Refresh-Ahead**: Proactively refresh cache before expiration

#### Cache Considerations
- **TTL (Time To Live)**: How long data stays cached
- **Cache Invalidation**: "There are only two hard things in Computer Science: cache invalidation and naming things"
- **Cache Warming**: Pre-populate cache with likely-needed data

## 8. Identifying Bottlenecks

### Common Bottlenecks
1. **Load Balancer**: Single point of failure
2. **Database**: Query performance, connection limits
3. **Network**: Bandwidth limitations between services
4. **Storage**: Disk I/O, especially for video processing
5. **CPU**: Intensive computations (ML inference, video processing)
6. **Memory**: Large datasets, caching layers

### Bottleneck Solutions

#### Load Balancer
- **Problem**: Single point of failure
- **Solution**: Multiple load balancers with DNS failover or floating IPs

#### Database
- **Read Bottleneck**: Read replicas, caching
- **Write Bottleneck**: Sharding, write-optimized databases
- **Connection Limits**: Connection pooling

#### Storage
- **Large Files**: Object storage (S3), CDN for delivery
- **High I/O**: SSD storage, database indexing

### Monitoring & Alerting
- **Application Metrics**: Request rate, error rate, latency
- **Infrastructure Metrics**: CPU, memory, disk, network
- **Business Metrics**: User engagement, conversion rates
- **Alert Thresholds**: P95 latency > 500ms, error rate > 1%

## 9. System Design Interview Process

### Step-by-Step Approach
1. **Clarify Requirements** (5 minutes)
   - Ask about scale, features, constraints
   - Define success metrics

2. **Estimate Scale** (5 minutes)
   - Calculate QPS, storage, bandwidth
   - Plan for growth

3. **High-Level Design** (15 minutes)
   - Draw main components
   - Define API contracts
   - Choose database type

4. **Detailed Design** (15 minutes)
   - Database schema
   - Specific technologies
   - Handle bottlenecks

5. **Address Scale & Reliability** (10 minutes)
   - Identify bottlenecks
   - Propose solutions
   - Discuss monitoring

### Common Mistakes to Avoid
- Don't jump into details without understanding requirements
- Don't ignore the scale - it drives architectural decisions
- Don't design for infinite scale if it's not needed
- Don't forget about monitoring and operational concerns
- Don't choose technologies without justification

## 10. Key Trade-offs

### Performance vs Scalability
- **Performance**: Optimize for small scale, fewer users
- **Scalability**: Design for growth, may sacrifice initial performance

### Latency vs Throughput
- **Low Latency**: Optimize for fast individual responses
- **High Throughput**: Optimize for handling many requests

### Availability vs Consistency
- **High Availability**: System stays up even during failures
- **Strong Consistency**: All users see the same data immediately
- **Eventual Consistency**: Data will be consistent eventually (good for availability)

### Cost vs Performance
- **Managed Services**: More expensive but less operational overhead
- **Self-Hosted**: Cheaper but requires more engineering effort

## 11. Technology Stack Examples

### High-Traffic Web Application
- **Load Balancer**: AWS ALB, NGINX
- **API Servers**: Node.js, Go, Java Spring Boot
- **Caching**: Redis, Memcached
- **Database**: PostgreSQL with read replicas
- **Message Queue**: RabbitMQ, Apache Kafka
- **Storage**: AWS S3 for files
- **Monitoring**: DataDog, New Relic, Prometheus

### Real-Time System (Chat App)
- **WebSocket Servers**: Socket.io, raw WebSockets
- **Message Broker**: Apache Kafka, Redis Pub/Sub
- **Database**: Redis for active sessions, PostgreSQL for history
- **Push Notifications**: Firebase, Apple Push Service

### Big Data Pipeline
- **Data Ingestion**: Apache Kafka, AWS Kinesis
- **Stream Processing**: Apache Storm, Spark Streaming
- **Batch Processing**: Apache Spark, Hadoop MapReduce
- **Storage**: HDFS, AWS S3, data lakes
- **Analytics**: Apache Druid, ClickHouse

## 12. Monitoring & Metrics

### SLA Metrics
- **Availability**: 99.9% uptime = 8.77 hours downtime/year
- **Latency**: P95 response time < 200ms
- **Throughput**: Handle 100,000 requests/second
- **Error Rate**: < 0.1% of requests fail

### Monitoring Best Practices
- Monitor at multiple levels (application, infrastructure, business)
- Set up alerting for SLA violations
- Use dashboards for real-time visibility
- Implement distributed tracing for microservices
- Log structured data for easier analysis

### Key Metrics to Track
- **RED Method**: Rate (requests/sec), Errors (error rate), Duration (latency)
- **USE Method**: Utilization, Saturation, Errors
- **Infrastructure**: CPU, memory, disk I/O, network
- **Business**: Active users, conversion rates, revenue

## 13. Common System Design Patterns

### Circuit Breaker
Prevents cascade failures by failing fast when downstream service is unhealthy.

### Rate Limiting
- **Token Bucket**: Allows bursts up to bucket size
- **Sliding Window**: Smooth rate limiting over time window
- **Fixed Window**: Simple but can allow double rate at window boundaries

### Bulkhead Pattern
Isolate critical resources so failure in one area doesn't affect others.

### CQRS (Command Query Responsibility Segregation)
Separate read and write operations for better performance and scalability.

### Event Sourcing
Store all changes as events, rebuild current state by replaying events.

## 14. Sample Interview Questions

### Easy
- Design a URL shortener
- Design a chat application
- Design a file storage system

### Medium
- Design Twitter/X
- Design Instagram
- Design a ride-sharing service
- Design a recommendation system

### Hard
- Design YouTube
- Design a search engine
- Design a distributed cache
- Design a payment system
- Design a data labeling platform (your example)

## 15. Pro Tips

### Communication
- Think out loud - explain your reasoning
- Draw diagrams - visual communication is powerful
- Ask clarifying questions throughout
- Acknowledge trade-offs explicitly

### Technical Depth
- Start simple, then add complexity
- Justify technology choices
- Consider operational aspects (deployment, monitoring)
- Think about failure scenarios

### Time Management
- Spend time understanding requirements
- Don't get stuck on one component
- Leave time for bottleneck discussion
- Practice with a timer

### Red Flags to Avoid
- Single points of failure without acknowledgment
- Choosing technologies without explanation
- Ignoring the given scale requirements
- Not considering data consistency requirements
- Forgetting about monitoring and operations

## Conclusion

System design interviews test your ability to think systematically about complex problems. The key is to:

1. **Understand the problem** through good questions
2. **Estimate scale** to drive decisions
3. **Design iteratively** from simple to complex
4. **Consider trade-offs** and justify choices
5. **Think operationally** about monitoring and maintenance

Remember: There's no single "correct" answer. Focus on demonstrating systematic thinking, technical knowledge, and the ability to make reasonable trade-offs while communicating clearly.