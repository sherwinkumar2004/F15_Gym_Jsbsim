# F15_Gym_Jsbsim
RL Environment used - Gym-JSBSim (https://github.com/Gor-Ren/gym-jsbsim, https://purehost.bath.ac.uk/ws/portalfiles/portal/216919613/Rennie_Gordon.pdf):
  1. Utilises JSBSim as the physics model.
  2. Provides the ability to visualise the training and testing process through FlightGear.
  3. The gym environment provides the ability to further modify the environment, and deploy different models and test their effectiveness.
  4. Presently two training environments are available.

Installation and other issues:
  1. ROOT_DIR variable in simulation.py of jsbsim folder should be modified to point towards the jsbsim installation locally
     
Further work:
  1. Verification metrics are yet to be added.
  2. Currently flightgear visualisation is turned off. Details about the visualisation are available in the gym-jsbsim repo.
