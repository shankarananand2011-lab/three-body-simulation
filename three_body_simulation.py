#this program takes in the masses, initial velocities and initial positions of three bodies to simulate their trajectories through time. it also demonstrates the system's sensitivity to initial conditions by plotting the natural log of the seperation of two systems with a very tiny difference in initial x_position against time.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
fig,ax=plt.subplots()
body_1,=ax.plot([],[],'ro')
body_2,=ax.plot([],[],'go')
body_3,=ax.plot([],[],'bo')
body_4, = ax.plot([], [], 'r^')
body_5, = ax.plot([], [], 'g^')
body_6, = ax.plot([], [], 'b^')
#these variables are the masses, positions and velocities of the three bodies
a = b = c = 1.0
x_p = [-1.0, 1.0, 0.0]
y_p = [0.0, 0.0, 1.0]
vx=[0.2, -0.2, 0.0]
vy=[0.0, 0.0, 0.3]
x_p2 = [-1.0+1e-6, 1.0, 0.0]
y_p2 = [0.0, 0.0, 1.0]
vx2 = [0.2, -0.2, 0.0]
vy2 = [0.0, 0.0, 0.3]
dt = 0.0001
steps = 5000
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
#this function takes in the x and y positions of two bodies to calculate their distance
def distance(x1,x2,y1,y2):
    return(((x1-x2)**2)+((y1-y2)**2))**.5
#this function takes in the masses and distance between two bodies to calculate their gravitational force
def gravitational_force(p,q,r):
    return((1.0)*(q*p)/(r**2))
#this function takes in the x and y positions, the distance and the gravitational force between two bodies to calculate the x and y components of that force
def components(x1,x2,y1,y2,d,f):
    return(f*(x2-x1)/d),(f*(y2-y1)/d)
#this function takes in the positions and masses of three bodies to calculate all their accelerations in x and y dimensions
def acc(x1, x2, x3, y1, y2, y3, mass1, mass2, mass3):
    r1=distance(x1,x2,y1,y2)
    r2=distance(x1,x3,y1,y3)
    r3=distance(x2,x3,y2,y3)
    f1=gravitational_force(mass1,mass2,r1)
    f2=gravitational_force(mass1,mass3,r2)
    f3=gravitational_force(mass2,mass3,r3)
    xf1,yf1=components(x1,x2,y1,y2,r1,f1)
    xf2,yf2=components(x1,x3,y1,y3,r2,f2)
    xf3,yf3=components(x2,x3,y2,y3,r3,f3)
    fx1 = xf1 + xf2
    fy1 = yf1 + yf2
    fx2 = -xf1 + xf3
    fy2 = -yf1 + yf3
    fx3 = -xf2 - xf3
    fy3 = -yf2 - yf3    
    return(fx1/mass1,fy1/mass1,fx2/mass2,fy2/mass2,fx3/mass3,fy3/mass3)
#this is the runge kutta function, which inputs the positions, velocities and masses of three bodies and uses the runge kutta method to output their positions and velocities after one frame. its more accurate compared to the euler method because it cleverly cancels out the second, thrid and fourth order derivatives of position.
def rk4(x1,x2,x3,y1,y2,y3,vx1,vx2,vx3,vy1,vy2,vy3,mass1,mass2,mass3):
    k1_ax1,k1_ay1,k1_ax2,k1_ay2,k1_ax3,k1_ay3=acc(x1,x2,x3,y1,y2,y3,mass1,mass2,mass3)
    k1_vx1,k1_vy1,k1_vx2,k1_vy2,k1_vx3,k1_vy3=vx1,vy1,vx2,vy2,vx3,vy3
    k2_ax1,k2_ay1,k2_ax2,k2_ay2,k2_ax3,k2_ay3=acc(x1+k1_vx1*dt/2,x2+k1_vx2*dt/2,x3+k1_vx3*dt/2,y1+k1_vy1*dt/2,y2+k1_vy2*dt/2,y3+k1_vy3*dt/2,mass1,mass2,mass3)
    k2_vx1,k2_vy1,k2_vx2,k2_vy2,k2_vx3,k2_vy3=vx1+k1_ax1*dt/2,vy1+k1_ay1*dt/2,vx2+k1_ax2*dt/2,vy2+k1_ay2*dt/2,vx3+k1_ax3*dt/2,vy3+k1_ay3*dt/2
    k3_ax1,k3_ay1,k3_ax2,k3_ay2,k3_ax3,k3_ay3=acc(x1+k2_vx1*dt/2,x2+k2_vx2*dt/2,x3+k2_vx3*dt/2,y1+k2_vy1*dt/2,y2+k2_vy2*dt/2,y3+k2_vy3*dt/2,mass1,mass2,mass3)
    k3_vx1,k3_vy1,k3_vx2,k3_vy2,k3_vx3,k3_vy3=vx1+k2_ax1*dt/2,vy1+k2_ay1*dt/2,vx2+k2_ax2*dt/2,vy2+k2_ay2*dt/2,vx3+k2_ax3*dt/2,vy3+k2_ay3*dt/2
    k4_ax1,k4_ay1,k4_ax2,k4_ay2,k4_ax3,k4_ay3=acc(x1+k3_vx1*dt,x2+k3_vx2*dt,x3+k3_vx3*dt,y1+k3_vy1*dt,y2+k3_vy2*dt,y3+k3_vy3*dt,mass1,mass2,mass3)
    k4_vx1,k4_vy1,k4_vx2,k4_vy2,k4_vx3,k4_vy3=vx1+k3_ax1*dt,vy1+k3_ay1*dt,vx2+k3_ax2*dt,vy2+k3_ay2*dt,vx3+k3_ax3*dt,vy3+k3_ay3*dt
    vx1=vx1+(k1_ax1+2*k2_ax1+2*k3_ax1+k4_ax1)*dt/6
    vy1=vy1+(k1_ay1+2*k2_ay1+2*k3_ay1+k4_ay1)*dt/6
    vx2=vx2+(k1_ax2+2*k2_ax2+2*k3_ax2+k4_ax2)*dt/6
    vy2=vy2+(k1_ay2+2*k2_ay2+2*k3_ay2+k4_ay2)*dt/6
    vx3=vx3+(k1_ax3+2*k2_ax3+2*k3_ax3+k4_ax3)*dt/6
    vy3=vy3+(k1_ay3+2*k2_ay3+2*k3_ay3+k4_ay3)*dt/6
    x1=x1+(k1_vx1+2*k2_vx1+2*k3_vx1+k4_vx1)*dt/6
    y1=y1+(k1_vy1+2*k2_vy1+2*k3_vy1+k4_vy1)*dt/6
    x2=x2+(k1_vx2+2*k2_vx2+2*k3_vx2+k4_vx2)*dt/6
    y2=y2+(k1_vy2+2*k2_vy2+2*k3_vy2+k4_vy2)*dt/6
    x3=x3+(k1_vx3+2*k2_vx3+2*k3_vx3+k4_vx3)*dt/6
    y3=y3+(k1_vy3+2*k2_vy3+2*k3_vy3+k4_vy3)*dt/6
    return x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3
traj_x1=[]
traj_x2=[]
traj_x3=[]
traj_y1=[]
traj_y2=[]
traj_y3=[]
#this function calls rk4 to calculate and update the positions and velocities of the bodies. it also calculates and stores the separation between the two systems
def update(frame):
    x_p[0],y_p[0],x_p[1],y_p[1],x_p[2],y_p[2],vx[0],vy[0],vx[1],vy[1],vx[2],vy[2]=rk4(x_p[0],x_p[1],x_p[2],y_p[0],y_p[1],y_p[2],vx[0],vx[1],vx[2],vy[0],vy[1],vy[2],a,b,c)
    x_p2[0],y_p2[0],x_p2[1],y_p2[1],x_p2[2],y_p2[2],vx2[0],vy2[0],vx2[1],vy2[1],vx2[2],vy2[2]=rk4(x_p2[0],x_p2[1],x_p2[2],y_p2[0],y_p2[1],y_p2[2],vx2[0],vx2[1],vx2[2],vy2[0],vy2[1],vy2[2],a,b,c)
    s1=distance(x_p2[0],x_p[0],y_p2[0],y_p[0])
    s2=distance(x_p2[1],x_p[1],y_p2[1],y_p[1])
    s3=distance(x_p2[2],x_p[2],y_p2[2],y_p[2])
    separations.append(max(s1,s2,s3))
    traj_x1.append(x_p[0])
    traj_y1.append(y_p[0])
    traj_x2.append(x_p[1])
    traj_y2.append(y_p[1])
    traj_x3.append(x_p[2])
    traj_y3.append(y_p[2])
    body_1.set_data([x_p[0]], [y_p[0]])
    body_2.set_data([x_p[1]], [y_p[1]])
    body_3.set_data([x_p[2]], [y_p[2]])
    body_4.set_data([x_p2[0]], [y_p2[0]])
    body_5.set_data([x_p2[1]], [y_p2[1]])
    body_6.set_data([x_p2[2]], [y_p2[2]])
    return body_1, body_2, body_3, body_4, body_5, body_6
separations=[]
ani=FuncAnimation(fig,update,frames=steps,interval=1,blit=True)
plt.show()
print(len(separations))
print(separations[:5])
plt.plot(np.log(separations))
plt.show()






























































































































































