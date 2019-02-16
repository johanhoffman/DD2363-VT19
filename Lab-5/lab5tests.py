import unittest
from lab5 import *

class Lab5FunctionsTest(unittest.TestCase):

    def tests(self):
        # pick an ODE of the form du/dt = a*u, a < 0
        #for explicit euler, solution is A-stable if |1+h*a| < 1
        f = lambda u_n,t: -3.0*u_n
        asol = lambda t: exp(-3.0*t)
        steps = [2,8,32,128]
        exact = asol(3.0)
        print("Convergence of scalar Explicit Euler")
        plt.figure(1)
        for s in steps:
            y,t,h = explicit_euler(3.0,f,1.0,s)
            print "Step size: " + str(h) + ", Error: " + str(abs(exact-y[-1]))
            plt.plot(t,y,label="STEP = %.3f" % (h))
        plt.legend()
        plt.title("Explicit Euler for 'du/dt = -3u'")
        plt.ylabel("y")
        plt.xlabel("t")


        print "\n\n\n"
        print("Convergence of Explicit Euler for Systems")
        plt.figure(2)
        # SYSTEM OF ODEs
        A = np.array([[2.,3.],[2.,1.]])
        # exact solutions
        y1 = lambda t: exp(-1.0*t) + 3.0*exp(4.0*t)
        y2 = lambda t: -1.0*exp(-1.0*t) + 2.0*exp(4.0*t)

        f = lambda y,t: A.dot(y)
        steps = [2,8,32,128]
        end_steps = None;
        for s in steps:
            y,t,h = explicit_euler_system(1.,f,np.array([4.,1.]),s)
            print "x: Step size: " + str(h) + ", Error: " + str(abs(y1(1.)-y[0,-1]))
            print "y: Step size: " + str(h) + ", Error: " + str(abs(y2(1.)-y[1,-1]))
            plt.subplot(121)
            plt.plot(t,y[0,:],label="step =  %.3f" % (h))
            plt.subplot(122)
            plt.plot(t,y[1,:],label="step =  %.3f" % (h))
            if s == steps[-1]:
                end_steps = t

        plt.subplot(121)
        plt.plot(t,[y1(x) for x in t],label="exact")
        plt.legend()
        plt.title("Explicit Euler for y1")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.subplot(122)
        plt.plot(t,[y2(x) for x in t],label="exact")
        plt.legend()
        plt.title("Explicit Euler for y2")
        plt.xlabel("t")
        plt.ylabel("y")

        plt.show()
if __name__ == '__main__':
    unittest.main()
