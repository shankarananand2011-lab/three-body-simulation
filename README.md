# three-body-simulation
Author:Shankaran Anand
The three body problem is a notoriously chaotic problem where a slight change in initial conditions can lead to a dramatic shift down the line. unlike the two body problem where the equal and opposite force of the bodies on each other is balanced by their perpendicular velocities, there is a constant tug of war between the objects where every body is being pulled by two bodies which are themselves constantly moving. a slight nudge or change in force can be enough to move a planet in a completely different direction.
This is a simulation that uses the runge kutta method to show the way the three body problem evolves through time for two slightly different initial conditions.
Requirements: Python 3, matplotlib, numpy
Install: pip install matplotlib numpy
Run: python three_body_simulation.py
The two systems are almost identical until a close gravitational encounter of the three bodies, after which bodies 1 and 2 grow apart considerably. the lyapunov exponent(ln of the seperation) itself stays low up until the encouter and increases steeply after the encounter and slows down.
The RK4 methoduses a clever trick of taking a weighted average of the velocities at the start, midpoint and end of a timestep to cancel out the second, third and fourth order derivatives, leading to dramatically more accuracy.
Update: softening parameter ε=0.1 added, energy conservation verified to 10⁻⁹ relative error, Lyapunov exponent extracted as λ=0.388 simulation_time⁻¹, mapping to SMBH scales gives Lyapunov time of 38,700 years for a 1.2×10⁷ solar mass triple system, aiming to change the parameters to distance instead of positions for n-dimensional analysis and stability mapping(to make dimensions a parameter)
. more updates to come
