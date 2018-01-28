file=open("data.txt", "a+")
file.close() 
number = 0
found = [0]
f=open("data.txt", "r")

if f.mode == 'r':
	
	contents =f.read()
for number in range(100):
	if(contents.find('H',number)==1) :
		found.append(int(number)
print(found)
file.close() 
