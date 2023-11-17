import numpy as np

## Function definition
# Trapezoid rule using loops
def trapezoidalLoop(function, aLimit, bLimit, numSubintervals):
    deltaX = (bLimit - aLimit) / numSubintervals
    currentInterval = 0.0
    # Performing calculation for fist edge in series
    currentInterval += function(aLimit) / 2.0
    # Iterating through series
    for i in range(1, numSubintervals):
        currentInterval += function(aLimit + i * deltaX)
    # Performing calculation for end edge subinterval
    currentInterval += function(bLimit) / 2.0
    trapLoopAnswer = currentInterval * deltaX
    return trapLoopAnswer

# Midpoint rule based approximation using a loop
def midpointLoop(function, aLimit, bLimit, numSubintervals):
    deltaX = (bLimit - aLimit) / numSubintervals
    midResult = 0
    for i in range(numSubintervals):
        midResult += function((aLimit + deltaX/2.0) + i*deltaX)
    midResult *= deltaX
    return midResult

# Trapezoid rule based approximation using vectorization
def trapezoidal(function, aLimit, bLimit, numSubintervals):
    deltaX = (bLimit - aLimit) / numSubintervals
    # All values in subintervals
    xArray = np.linspace(aLimit, bLimit, numSubintervals)

    integralFunction = function(xArray)
    # Apply the function with variable
    iTrapezoid = (np.sum(integralFunction) - 0.5*function(aLimit) - 0.5*function(bLimit)) * deltaX
    return iTrapezoid

# Midpoint rule based approximation
def midpoint(function, aLimit, bLimit, numSubintervals):
    deltaX = (bLimit - aLimit) / numSubintervals
    # All intervals in subintervals
    xArray = np.linspace((aLimit + (deltaX/2)), (bLimit - (deltaX/2)), numSubintervals)
    integralFunction = function(xArray)
#    integrateR = xArray[:numSubintervals - 1]
#    integrateL = xArray[1:]
    iMidline = deltaX * np.sum(integralFunction)

    return(iMidline)

# Main Program
def main():
    # Testing trapezoid function
    integralEquation = lambda x: 1 / x
    subintervalN = [2, 10, 50, 250]

    # Testing functionality for vectorized trapezoid approximation
    errorsTrap = []
    integralListTrap = []
    # Running Integrals
    for i in subintervalN:
        integralListTrap.append(trapezoidal(integralEquation, 1, 2, i))

    # Error Calculation
    for j in range(0, len(integralListTrap)):
        errorsTrap.append(np.log(2) - integralListTrap[j])

    # Ratio Calculation
    ratioTrap = []
    for i in range(0 , len(errorsTrap)-1):
        ratioCurrent = errorsTrap[i] / errorsTrap[i+1]
        ratioTrap.append(ratioCurrent)
    # Doing final calculation without going out of range
    ratioTrap.append(errorsTrap[len(errorsTrap) - 1] / errorsTrap[len(errorsTrap) - 2])
    
    print()
    print("Approximations of Integral with trapezoidal rule")
    for j in range(0,len(subintervalN)):
        print("Subintervals(n): {0:3}  Estimate: {1:8.7f} Error: {2:8.7f} Ratio of errors: {3:.3f}".format(subintervalN[j], integralListTrap[j],errorsTrap[j],ratioTrap[j]))
    print()

    # Testing Midpoint formula approximation
    errorMid = []
    integralListMid = []
    print("Approximations of Integral with midpoint rule")
    for i in subintervalN:
        integralListMid.append(midpoint(integralEquation, 1, 2, i))

    # Error Calculation
    for j in range(0, len(integralListMid)):
        errorMid.append(np.log(2) - integralListMid[j])

    # Ratio Calculation
    ratioMid = []
    for i in range(0 , len(errorMid)-1):
        ratioCurrent = errorMid[i] / errorMid[i+1]
        ratioMid.append(ratioCurrent)

    # Appending on final ratio
    ratioMid.append(errorMid[len(errorMid) - 1] / errorMid[len(errorMid) - 2])

    for j in range(0,len(subintervalN)):
        print("Subintervals(n): {0:3}  Estimate: {1:8.7f} Error: {2:8.7f} Ratio of error: {3:.3f}".format(subintervalN[j], integralListMid[j],errorMid[j],ratioMid[j]))

    print()

    # Testing trapezoid loop series
    errorTrapLoop = []
    integralListTrapLoop = []
    # Running Integrals
    for i in subintervalN:
        integralListTrapLoop.append(trapezoidalLoop(integralEquation, 1, 2, i))

    # Error Calculation
    for j in range(0, len(integralListTrapLoop)):
        errorTrapLoop.append(np.log(2) - integralListTrapLoop[j])

    # Ratio Calculation)
    ratioTrapLoop = []
    for i in range(0 , len(errorTrapLoop)-1):
        ratioCurrent = errorTrapLoop[i] / errorTrapLoop[i+1]
        ratioTrapLoop.append(ratioCurrent)
    # Doing final calculation without going out of range
    ratioTrapLoop.append(errorTrapLoop[len(errorTrapLoop) - 1] / errorTrapLoop[len(errorTrapLoop) - 2])
   
    print("Approximations of Integral with trapezoidal rule, loop version")
    for j in range(0,len(subintervalN)):
        print("Subintervals(n): {0:3}  Estimate: {1:8.7f} Error: {2:8.7f} Ratio of error: {3:.3f}".format(subintervalN[j], integralListTrapLoop[j],errorTrapLoop[j],ratioTrapLoop[j]))
    print()

    # Testing Midpoint loop formula approximation
    errorMidLoop = []
    integralListMidLoop = []
    print("Approximations of Integral with midpoint rule, loop version")
    for i in subintervalN:
        integralListMidLoop.append(midpointLoop(integralEquation, 1, 2, i))

    # Error Calculation
    for j in range(0, len(integralListMidLoop)):
        errorMidLoop.append(np.log(2) - integralListMidLoop[j])

    # Ratio Calculation
    ratioMidLoop = []
    for i in range(0 , len(errorMidLoop)-1):
        ratioCurrent = errorMidLoop[i] / errorMidLoop[i+1]
        ratioMidLoop.append(ratioCurrent)

    # Appending on final ratio
    ratioMidLoop.append(errorMidLoop[len(errorMidLoop) - 1] / errorMidLoop[len(errorMidLoop) - 2])

    for j in range(0,len(subintervalN)):
        print("Subintervals(n): {0:3}  Estimate: {1:8.7f} Error: {2:8.7f} Ratio of error: {3:.3f}".format(subintervalN[j], integralListMidLoop[j],errorMidLoop[j],ratioMidLoop[j]))


    """Commentary on error ratio reporting

    The ratio of the errors shows how by increasing the amount of
    subintervals being used will increase our accuracy of our
    approximation for the integral.

    """
#    print("Error List Trapezoid",errorsTrap)
#    print()
#    print("Error List Trapezoid Loop",errorTrapLoop)
#    print()
#    print("Error List Midline Loop",errorMid)
#    print()
#    print("Error List Midline Loop",errorMidLoop)
#    print()
#
#    print("Ratio Trap", ratioTrap)
#    print()
#    print("Ratio Trap Loop", ratioTrapLoop)
#    print()
#    print("Ratio Mid", ratioMid)
#    print()
#    print("Ratio Mid Loop", ratioMidLoop)
#    print()

if __name__ == "__main__":
    main()
