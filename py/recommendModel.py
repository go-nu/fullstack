import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="1234"
    )
    print("✅ MySQL 연결 성공")
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    for db in cursor.fetchall():
        print("📁", db[0])
    conn.close()
except mysql.connector.Error as err:
    print("❌ MySQL 연결 실패:", err)
