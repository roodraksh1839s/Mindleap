import mysql.connector 
connector = mysql.connector.connect(host="localhost",user="root",password="superb$1839S",database="college")
print("Database seelected")
cursor = connector.cursor()
cursor.execute("""insert into student(ST_ID, STUDENT_NAME, MOBILE_NO,DEPARTMENT)
               values(3, "SAMEER","89898989","CSE");""")
connector.commit()
cursor.close()
connector.close()
print("Entered the data sucessfully")
#This is the our first trial of version control