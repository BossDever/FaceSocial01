#!/bin/bash

echo "=== FaceSocial Debug Information ==="
echo

echo "1. ตรวจสอบ Frontend container:"
docker logs facesocial_frontend_dev --tail 20

echo
echo "2. ตรวจสอบ Database connection:"
docker exec facesocial_postgres_dev psql -U admin -d facesocial -c "SELECT COUNT(*) as user_count FROM users;"

echo
echo "3. ตรวจสอบ Face API:"
curl -s http://localhost:8080/health | head -5

echo
echo "4. ตรวจสอบ Container status:"
docker ps --filter "name=facesocial" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "5. ตรวจสอบ Frontend health:"
curl -s http://localhost:3000/api/health || echo "Frontend API not responding"

echo
echo "=== Debug Complete ==="
