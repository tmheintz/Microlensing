# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:23:21 2017

@author: tmhei
"""
from __future__ import division, print_function, unicode_literals
import numpy
import math
import sys
from matplotlib import pyplot

class Image:
    
    def __init__(self):
        self.points = []
        self.H = None
        self.A = 0
        self.i = complex(0,1)
        
        self.indexes_zeta = []
        
    
    def find_A(self, event, zeta, center, dtheta):
         ## This is the Trapezium approximation with a parabolic correction (Bozza 2010)
        
        A_trap = numpy.zeros(len(self.points))
        A_par = numpy.zeros(len(self.points))
        partial_zeta_partial_zbar_i= numpy.zeros(len(self.points), dtype = complex)
        partial_zeta_partial_zbar_i1= numpy.zeros(len(self.points), dtype = complex)
        partial2_zetabar_partial_z2_i= numpy.zeros(len(self.points), dtype = complex)
        partial2_zetabar_partial_z2_i1= numpy.zeros(len(self.points), dtype = complex)
        zeta_prime_i = numpy.zeros(len(self.points), dtype = complex)        
        zeta_prime_i1 = numpy.zeros(len(self.points), dtype = complex)        
        
        
        ## Trapezium Approximation ##
        
        A_trap[:len(self.points)-1] = 1/2*(numpy.imag(self.points[1:]) + numpy.imag(self.points[:len(self.points)-1]))*(-1*numpy.real(self.points[1:]) + numpy.real(self.points[:len(self.points)-1]))
        A_trap[-1] = 1/2*(self.points[0].imag + self.points[-1].imag)*(-self.points[0].real + self.points[-1].real)
        
        ## Parabolic Correction ##
        
        zeta_prime_i = numpy.subtract(zeta[self.indexes_zeta][:], center)*self.i
        zeta_prime_i1[:len(self.points)-1] = numpy.subtract(zeta[self.indexes_zeta][1:], center)*self.i
        zeta_prime_i1[-1] = numpy.subtract(zeta[self.indexes_zeta][0], center)*self.i
        
        ## For these, I define the planets position opposite of Bozza 2010 so switch signs when using a/2
        
        partial_zeta_partial_zbar_i = 1/(1 + event.q)*(
                                1/numpy.square(numpy.subtract(numpy.conj(self.points[:]), event.a/2.0)) 
                                + event.q/numpy.square(numpy.add(numpy.conj(self.points[:]), event.a/2.0))
                                                        )
        partial_zeta_partial_zbar_i1[:len(self.points)-1] = 1/(1 + event.q)*(
                                    1/numpy.square(numpy.subtract(numpy.conj(self.points[1:]), event.a/2.0)) 
                                     + event.q/numpy.square(numpy.add(numpy.conj(self.points[1:]), event.a/2.0))
                                                                             )
        partial_zeta_partial_zbar_i1[-1] = 1/(1 + event.q)*(
                                            1/(numpy.subtract(numpy.conj(self.points[0]), event.a/2.0))**2 
                                            + event.q/(numpy.add(numpy.conj(self.points[0]), event.a/2.0))**2
                                                              )
        
        
        partial2_zetabar_partial_z2_i = -2/(1+event.q)*(
                                            1/numpy.power(numpy.subtract(self.points[:], event.a/2.0), 3) 
                                            + event.q/numpy.power(numpy.add(self.points[:], event.a/2.0), 3)
                                                        )
        partial2_zetabar_partial_z2_i1[:len(self.points)-1] = -2/(1+event.q)*(
                                        1/numpy.power(numpy.subtract(self.points[1:], event.a/2.0), 3) 
                                        + event.q/numpy.power(numpy.add(self.points[1:], event.a/2.0), 3)
                                                                                )
        partial2_zetabar_partial_z2_i1[-1] = -2/(1+event.q)*(
                                1/(numpy.subtract(self.points[0], event.a/2.0))**3 
                                  + event.q/(numpy.add(self.points[0], event.a/2.0))**3
                                                                )

        
        Ji = 1 - numpy.square(numpy.absolute(partial_zeta_partial_zbar_i))
        Ji1 = 1 - numpy.square(numpy.absolute(partial_zeta_partial_zbar_i1))
        z_prime_i = (zeta_prime_i - partial_zeta_partial_zbar_i*numpy.conj(zeta_prime_i))/Ji            
        z_prime_i1 = (zeta_prime_i1 - partial_zeta_partial_zbar_i1*numpy.conj(zeta_prime_i1))/Ji1

        A_par = 1/24.0*(
            (event.rho**2 + numpy.imag(numpy.square(z_prime_i)*zeta_prime_i*partial2_zetabar_partial_z2_i))/Ji 
            + (event.rho**2 + numpy.imag(numpy.square(z_prime_i1)*zeta_prime_i1*partial2_zetabar_partial_z2_i1))/Ji1
                      )*dtheta**3


        self.A = math.fabs(numpy.sum(A_trap) + numpy.sum(A_par))

    def plot_image(self, axes, color = (1,0,0)):
        for i in range(len(self.points)):
            axes.plot(self.points[i].real, self.points[i].imag, ".", color = color, markersize = 0.5)
    
    
class MicroBinaryEvent_ps:
    
    def __init__(self, q, a, tE, u0, alpha = 0, t0 = 0, npoints=1e5):
        ## initialize the parameters of the event
        
        ## q is the mass ratio between the two masses, a is seperation of the two masses, and 
        ## epsilon1 and epsilon 2 are mass ratios relative to total mass
        ## u0 is impact parameter, alpha is angle of trajectory relative to the line connecting the two masses
        ## t0 is the time when the lens and source are at closest point to each other in plane of the sky
        ## tE is the Einstein time (Einstein ring crossing time)
        ## npoints is used to determine amount of total points for the maps (i.e. if 100 is chosen then the map will calculate a 10x10 grid)

        self.q = q
        self.a = a
        self.u0 = u0
        self.alpha = alpha
        self.tE = tE
        self.t0 = t0
        
        self.z1 = self.a/2.0 + 0j     ## will treat the lenses as if they are on x-axis both a/2 away from origin
        self.epsilon1 = 1/(1+self.q)
        self.epsilon2 = 1-self.epsilon1
        self.m = (self.epsilon1 + self.epsilon2)/2.0
        self.dm = (self.epsilon2 - self.epsilon1)/2.0
        
        ## These will be used to make maps in Source and Lens Plane
        ## The maps will be from -(2+a/2) to (2+a/2)
        ## Edit the xvals to make maps bigger or larger.  Probably should add parameter to choose this but
        ## can also write MBE.xvals = ... to change but then you will also have to change the yvals as well
        
        self.npoints = npoints
        self.xvals = numpy.arange(-self.a/2. - 2, self.a/2.+2, 2*(self.a/2. + 2)/math.sqrt(self.npoints))
        self.yvals = self.xvals
        self.AMap_SourcePlane = numpy.zeros( (len(self.yvals), len(self.xvals)) )
        self.AMap_LensPlane = numpy.zeros( (len(self.yvals), len(self.xvals)) )
        
        ## These are used for the critical curves and caustics ##
        self.crit_pointsx = []
        self.crit_pointsy = []
        self.caustic_pointsx = []
        self.caustic_pointsy = []
        
        self.plot_images = False
        figure, axes = pyplot.subplots()
        self.axes = axes
        
    def calc_Mag(self, t, images = None):
        ## Calculate the Magnification at time t relative to t0 in units of einstein time for a point source
        ## You can pass a list for the images and as long as it is passed by reference it will be changed at the end of this block of code
        
        if images is None:
            images = []
        t = t/self.tE - self.t0/self.tE
        zeta = complex(math.cos(self.alpha)*t + self.u0*math.sin(self.alpha) + self.z1, self.u0*math.cos(self.alpha) - math.sin(self.alpha)*t)
        p = calc_Coeff(self.z1, zeta, self.dm, self.m)
        all_images = numpy.roots(p)
        
        for k in range(len(all_images)):
            val = all_images[k] - zeta + self.epsilon1/(self.z1 - all_images[k].conjugate()) + self.epsilon2/(-self.z1 - all_images[k].conjugate())
            if numpy.absolute(val) <= 1e-9:
                images.append(all_images[k])
        
        A = 0.0
        for j in range(len(images)):
            detJ = 1 - (self.epsilon1/(images[j].conjugate() - self.z1)**2 + self.epsilon2/(images[j].conjugate() + self.z1)**2)*(self.epsilon1/(images[j] - self.z1)**2 + self.epsilon2/(images[j] + self.z1)**2)
            A += abs(1/detJ)
        
        if self.plot_images:
            for k in range(len(images)):
                self.axes.plot(images[k].real, images[k].imag, "b.")
            
        return math.fabs(A)
    
    def calcSourceMagMap(self):
        ## Calculate the source plane magnification map for point source
        ## It is stored in the object itself so you just need to run this once
        
        for ix in range(len(self.xvals)):
            for iy in range(len(self.yvals)):
                zeta = complex(self.xvals[ix],self.yvals[iy])
                p = calc_Coeff(self.z1, zeta, self.dm, self.m)
                images = numpy.roots(p)  ## Can use any root solver here. I would recommend Skowron and Gould's root finder 
                mag = 0
                for k in range(len(images)):
                    z = images[k]
                    if images[k] - zeta + self.epsilon1/(self.z1 - images[k].conjugate()) + self.epsilon2/(-self.z1 - images[k].conjugate()) <= 1e-7 and images[k] - zeta + self.epsilon1/(self.z1 - images[k].conjugate()) + self.epsilon2/(-self.z1 - images[k].conjugate()) >= -1e-7:
                        detJ = 1 - (self.epsilon1/(z.conjugate() - self.z1)**2 + self.epsilon2/(z.conjugate() + self.z1)**2)*(self.epsilon1/(z - self.z1)**2 + self.epsilon2/(z + self.z1)**2)
                        if detJ == 0: ## don't want to divide by zero
                            detJ = 1e-15
                        mag +=  abs(1/detJ)
                    
                self.AMap_SourcePlane[iy,ix] = mag 
       
        
    def calcLensMagMap(self):
        ## Calculate the lens plane magnification map for point source
        ## It is stored in the object itself so you just need to run this once
        
        for ix in range(len(self.xvals)):
            for iy in range(len(self.yvals)):
                z = complex(self.xvals[ix],self.yvals[iy])
                detJ = 1 - (self.epsilon1/(z.conjugate() - self.z1)**2 + self.epsilon2/(z.conjugate() + self.z1)**2)*(self.epsilon1/(z - self.z1)**2 + self.epsilon2/(z + self.z1)**2)
                if detJ == 0:
                    detJ = 1e-15
                self.AMap_LensPlane[iy,ix] = abs(1/detJ)
        
    def calcCritPointsandCaustics(self):
        ## Calculate the Critical Points in the lens plane and the Caustics in the Source Plane
        ## Only need to run this once and then the caustics and crit points will be saved.
        ## Make sure to run again if switching parameters but using the same instance of the object
        
        phi = numpy.arange(0,2*math.pi,math.pi/1000.)
        crit_curves = numpy.zeros(len(phi), dtype = object)
        for i in range(len(phi)):
            c4 = 1
            c3 = 0
            c2 = -(2*self.z1**2 + complex(math.cos(phi[i]), math.sin(phi[i])))
            c1 = -2*self.z1*complex(math.cos(phi[i]), math.sin(phi[i]))*(self.epsilon1 - self.epsilon2)
            c0 = self.z1**2*(self.z1**2 - complex(math.cos(phi[i]), math.sin(phi[i])))
            crit_curves[i] = numpy.roots(numpy.array([c4,c3,c2, c1, c0]))
        for i in range(len(crit_curves)):
            points = crit_curves[i]
            for j in range(len(points)):                    
                self.crit_pointsx.append(points[j].real)
                self.crit_pointsy.append(points[j].imag)
        
        for i in range(len(self.crit_pointsx)):
            z = complex(self.crit_pointsx[i],self.crit_pointsy[i])
            zeta = z + self.epsilon1/(self.z1 - z.conjugate()) + self.epsilon2/(-self.z1 - z.conjugate())
            self.caustic_pointsx.append(zeta.real)
            self.caustic_pointsy.append(zeta.imag)

# ======================================================================
def calc_Coeff(z1, zeta, dm, m):
    ## Coefficients used for polynomial in z to find roots to find image positions
    ## Coefficients are taken from Witt and Mao(1995)
    
    zeta_bar = zeta.conjugate()
    c0 = z1**2*(4*dm**2*zeta + 4*m*dm*z1 + 4*dm*zeta*zeta_bar*z1 + 2*m*zeta_bar*z1**2 + zeta*zeta_bar**2*z1**2 - 2*dm*z1**3 - zeta*z1**4)
    c1 = -8*m*dm*zeta*z1 - 4*dm**2*z1**2 - 4*m**2*z1**2 - 4*m*zeta*zeta_bar*z1**2 - 4*dm*zeta_bar*z1**3 - zeta_bar**2*z1**4 + z1**6
    c2 = 4*m**2*zeta + 4*m*dm*z1 - 4*dm*zeta*zeta_bar*z1 - 2*zeta*zeta_bar**2*z1**2 + 4*dm*z1**3 + 2*zeta*z1**4
    c3 = 4*m*zeta*zeta_bar + 4*dm*zeta_bar*z1 + 2*zeta_bar**2*z1**2 - 2*z1**4
    c4 = -2*m*zeta_bar + zeta*zeta_bar**2 - 2*dm*z1 - zeta*z1**2
    c5 = z1**2 - zeta_bar**2
    
    return numpy.array([c5,c4,c3,c2,c1,c0])


# ======================================================================
#################Starting Finite Source Class#########################
# ======================================================================



class MicroBinaryEvent_fs:
    
    def __init__(self, q, a, tE, u0, rho, alpha = 0, t0 = 0, npoints=1e5):
        ## initialize the parameters of the event
        
        ## q is the mass ratio between the two masses, a is seperation of the two masses, and 
        ## epsilon1 and epsilon 2 are mass ratios relative to total mass
        ## u0 is impact parameter, alpha is angle of trajectory relative to the line connecting the two masses
        ## rho is the radius of the source
        ## t0 is the time when the lens and source are at closest point to each other in plane of the sky
        ## tE is the Einstein time (Einstein ring crossing time)
        ## npoints is used to determine amount of total points for the maps (i.e. if 100 is chosen then the map will calculate a 10x10 grid)

        self.q = q
        self.a = a
        self.u0 = u0
        self.rho = rho
        self.boundary = numpy.zeros(100, dtype = object)
        self.dtheta = 2*math.pi/len(self.boundary)
        for i in range(len(self.boundary)):
            phi = 2*i*math.pi/len(self.boundary)
            self.boundary[i] = complex(math.cos(phi), math.sin(phi))
        self.alpha = alpha
        self.tE = tE
        self.t0 = t0
        
        self.z1 = self.a/2.0 + 0j     ## will treat the lenses as if they are on x-axis both a/2 away from origin
        self.epsilon1 = 1/(1+self.q)
        self.epsilon2 = 1-self.epsilon1
        self.m = (self.epsilon1 + self.epsilon2)/2.0
        self.dm = (self.epsilon2 - self.epsilon1)/2.0
        
        ## These will be used to make maps in Source and Lens Plane
        ## The maps will be from 
        ## Edit the xvals to make maps bigger or larger.  Probably should add parameter to choose this but
        ## can also write MBE.xvals = ... to change but then you will also have to change the yvals as well
        self.npoints = npoints
        self.xvals = numpy.arange(-self.a/2. - 2, self.a/2.+2, 2*(self.a/2. + 2)/math.sqrt(self.npoints))
        self.yvals = self.xvals
        self.AMap_SourcePlane = numpy.zeros( (len(self.yvals), len(self.xvals)) )
        self.AMap_LensPlane = numpy.zeros( (len(self.yvals), len(self.xvals)) )
        
        ## These are used for the critical curves and caustics ##
        self.crit_pointsx = []
        self.crit_pointsy = []
        self.caustic_pointsx = []
        self.caustic_pointsy = []
        
        self.plot_images = False
        figure, axes = pyplot.subplots()
        self.axes = axes
        self.reject_vals = []
        self.vals = []
        self.find_cutoff = True
        
    def calc_Mag(self, t):
        ## Calculate the Magnification at time t relative to t0 in units of einstein time for a finite source
        ## You can pass a list for the images and as long as it is passed by reference it will be changed at the end of this block of code
        images = numpy.array([])
        t = t/self.tE - self.t0/self.tE
        zeta = complex(math.cos(self.alpha)*t + self.u0*math.sin(self.alpha) + self.z1, self.u0*math.cos(self.alpha) - math.sin(self.alpha)*t)
        pos_array = zeta + self.rho*self.boundary
        for i in range(len(pos_array)):
            point = pos_array[i]    
            p = calc_Coeff(self.z1, point, self.dm, self.m)
            all_images = numpy.roots(p)
            
            ## Check to see if the roots satisfy the lens equation
            
            if self.find_cutoff:
                val_array = numpy.absolute((numpy.conjugate(all_images) - self.z1)*(numpy.conjugate(all_images) + self.z1)*(all_images - point) - self.epsilon1*(
                        numpy.conjugate(all_images) + self.z1) - self.epsilon2*(numpy.conjugate(all_images) - self.z1))
                ## find cutoff
                r1 = numpy.max(val_array)
                val_array1 = numpy.delete(val_array, numpy.where(val_array == r1))
                r2 = numpy.max(val_array1)
                val_array2 = numpy.delete(val_array1, numpy.where(val_array1 == r2))

                r3 = numpy.max(val_array2)
                if math.fabs(numpy.log10(numpy.absolute(r2)) - numpy.log10(numpy.absolute(r3))) < 1e-2:
                    self.cutoff = 10**(math.log10(numpy.absolute(r1)) - math.fabs(numpy.log10(numpy.absolute(r2)) - numpy.log10(numpy.absolute(r1)))/2)
                else:
                    self.cutoff = 10**(math.log10(numpy.absolute(r2)) - math.fabs(numpy.log10(numpy.absolute(r2)) - numpy.log10(numpy.absolute(r3)))/2)

                self.find_cutoff = False
                
            
            for k in range(len(all_images)):
                val = (all_images[k].conjugate() - self.z1)*(all_images[k].conjugate() + self.z1)*(all_images[k] - point) - self.epsilon1*(
                        all_images[k].conjugate() + self.z1) - self.epsilon2*(all_images[k].conjugate() - self.z1)

                if numpy.absolute(val) <= self.cutoff:
                    if i == 0:
                        images = numpy.append(images, Image())
                        images[-1].points.append(all_images[k])
                        images[-1].indexes_zeta.append(i)
                    else:
                        j = self.find_closest_image(images, all_images[k])
                        images[j].points.append(all_images[k])
                        images[j].indexes_zeta.append(i)
            
        A = 0.0
        ## Calculate areas of images
        for j in range(len(images)):
            images[j].find_A(self, pos_array, zeta, self.dtheta)
            ## Add up the areas  
            A += images[j].A
        if self.plot_images:
            for k in range(len(pos_array)):
                self.axes.plot(pos_array[i].real, pos_array[j].imag,"g.")
            pyplot.show()
        
        return A/(math.pi*self.rho**2)
        
    
    def calcSourceMagMap(self):
        ## Calculate the source plane magnification map for point source
        ## It is stored in the object itself so you just need to run this once
        
        for ix in range(len(self.xvals)):
            for iy in range(len(self.yvals)):
                zeta = complex(self.xvals[ix],self.yvals[iy])
                p = calc_Coeff(self.z1, zeta, self.dm, self.m)
                images = numpy.roots(p)  ## Can use any root solver here I would recommend Skowron and Gould's root finder 
                mag = 0
                for k in range(len(images)):
                    z = images[k]
                    if images[k] - zeta + self.epsilon1/(self.z1 - images[k].conjugate()) + self.epsilon2/(-self.z1 - images[k].conjugate()) <= 1e-7 and images[k] - zeta + self.epsilon1/(self.z1 - images[k].conjugate()) + self.epsilon2/(-self.z1 - images[k].conjugate()) >= -1e-7:
                        detJ = 1 - (self.epsilon1/(z.conjugate() - self.z1)**2 + self.epsilon2/(z.conjugate() + self.z1)**2)*(self.epsilon1/(z - self.z1)**2 + self.epsilon2/(z + self.z1)**2)
                        if detJ == 0: ## don't want to divide by zero
                            detJ = 1e-15
                        mag +=  abs(1/detJ)
                    
                self.AMap_SourcePlane[iy,ix] = mag 
       
        
    def calcLensMagMap(self):
        ## Calculate the lens plane magnification map for point source
        ## It is stored in the object itself so you just need to run this once
        
        for ix in range(len(self.xvals)):
            for iy in range(len(self.yvals)):
                z = complex(self.xvals[ix],self.yvals[iy])
                detJ = 1 - (self.epsilon1/(z.conjugate() - self.z1)**2 + self.epsilon2/(z.conjugate() + self.z1)**2)*(self.epsilon1/(z - self.z1)**2 + self.epsilon2/(z + self.z1)**2)
                if detJ == 0:
                    detJ = 1e-15
                self.AMap_LensPlane[iy,ix] = abs(1/detJ)
        
    def calcCritPointsandCaustics(self):
        ## Calculate the Critical Points in the lens plane and the Caustics in the Source Plane
        ## Only need to run this once and then the caustics and crit points will be saved.
        ## Make sure to run again if switching parameters but using the same instance of the object
        
        phi = numpy.arange(0,2*math.pi,math.pi/1000.)
        crit_curves = numpy.zeros(len(phi), dtype = object)
        for i in range(len(phi)):
            c4 = 1
            c3 = 0
            c2 = -(2*self.z1**2 + complex(math.cos(phi[i]), math.sin(phi[i])))
            c1 = -2*self.z1*complex(math.cos(phi[i]), math.sin(phi[i]))*(self.epsilon1 - self.epsilon2)
            c0 = self.z1**2*(self.z1**2 - complex(math.cos(phi[i]), math.sin(phi[i])))
            crit_curves[i] = numpy.roots(numpy.array([c4,c3,c2, c1, c0]))
        for i in range(len(crit_curves)):
            points = crit_curves[i]
            for j in range(len(points)):                    
                self.crit_pointsx.append(points[j].real)
                self.crit_pointsy.append(points[j].imag)
        
        for i in range(len(self.crit_pointsx)):
            z = complex(self.crit_pointsx[i],self.crit_pointsy[i])
            zeta = z + self.epsilon1/(self.z1 - z.conjugate()) + self.epsilon2/(-self.z1 - z.conjugate())
            self.caustic_pointsx.append(zeta.real)
            self.caustic_pointsy.append(zeta.imag)
            
    
    def find_closest_image(self, images, point):
        ## Finds closest image to the point specified
        
        dist = 1000
        for i in range(len(images)):
            new_dist = abs(point - images[i].points[-1])
            if new_dist < dist:
                dist = new_dist
                j = i
        return j