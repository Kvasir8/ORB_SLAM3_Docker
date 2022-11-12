import math
import numpy as np

class GPSutils:

    def __init__(self):

        self.a = 6378137.0   # WGS-84 Earth semimajor axis (m)
        self.b = 6356752.314245 # Derived Earth semiminor axis (m)
        
        self.f = (self.a - self.b) / self.a
        self.f_inv = 1.0/self.f

        self.a_sq = self.a * self.a
        self.b_sq = self.b * self.b
        self.e_sq = self.f * (2 -self.f)

    def GeodeticToEcef(self, lat, lon, h):

        lamda = math.radians(lat)
        phi = math.radians(lon)
        s = math.sin(lamda)
        N = self.a / math.sqrt(1 - self.e_sq * s * s)

        sin_lambda = math.sin(lamda)
        cos_lambda = math.cos(lamda)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        x = (h + N) * cos_lambda * cos_phi
        y = (h + N) * cos_lambda * sin_phi
        z = (h + (1 - self.e_sq) * N) * sin_lambda

        return [x, y, z]

    def EcefToGeodetic(self, x, y, z):

        eps = self.e_sq / (1.0 - self.e_sq)
        p = math.sqrt(x * x + y * y)
        q = math.atan2((z * self.a), (p * self.b))
        sin_q = math.sin(q)
        cos_q = math.cos(q)
        sin_q_3 = sin_q * sin_q * sin_q
        cos_q_3 = cos_q * cos_q * cos_q
        phi = math.atan2((z + eps * self.b * sin_q_3), (p - self.e_sq * self.a * cos_q_3))
        lamda = math.atan2(y, x)
        v = self.a / math.sqrt(1.0 - self.e_sq * math.sin(phi) * math.sin(phi))
        h = (p / math.cos(phi)) - v

        lat = math.degrees(phi)
        lon = math.degrees(lamda)

        return [lat, lon, h]

    def EcefToEnu(self, x, y, z, lat0, lon0, h0):

        lamda = math.radians(lat0)
        phi = math.radians(lon0)
        s = math.sin(lamda)
        N = self.a / math.sqrt(1 - self.e_sq * s * s)

        sin_lambda = math.sin(lamda)
        cos_lambda = math.cos(lamda)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        x0 = (h0 + N) * cos_lambda * cos_phi;
        y0 = (h0 + N) * cos_lambda * sin_phi;
        z0 = (h0 + (1 - self.e_sq) * N) * sin_lambda
        
        xd = x - x0;
        yd = y - y0;
        zd = z - z0;

        ## This is the matrix multiplication

        xEast = -sin_phi * xd + cos_phi * yd
        yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
        zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

        return [xEast, yNorth, zUp]

    def EnuToEcef(self, xEast, yNorth, zUp, lat0, lon0, h0):

        lamda = math.radians(lat0);
        phi = math.radians(lon0);
        s = math.sin(lamda);
        N = self.a / self.sqrt(1 - self.e_sq * s * s);

        sin_lambda = math.sin(lamda);
        cos_lambda = math.cos(lamda);
        cos_phi = math.cos(phi);
        sin_phi = math.sin(phi);

        x0 = (h0 + N) * cos_lambda * cos_phi;
        y0 = (h0 + N) * cos_lambda * sin_phi;
        z0 = (h0 + (1 - self.e_sq) * N) * sin_lambda;

        xd = -sin_phi * xEast - cos_phi * sin_lambda * yNorth + cos_lambda * cos_phi * zUp;
        yd = cos_phi * xEast - sin_lambda * sin_phi * yNorth + cos_lambda * sin_phi * zUp;
        zd = cos_lambda * yNorth + sin_lambda * zUp;

        x = xd + x0;
        y = yd + y0;
        z = zd + z0;

        return[x, y, z]

    def GeodeticToEnu(self, lat, lon, h, lat0, lon0, h0):

        x, y, z = self.GeodeticToEcef(lat, lon, h)
        xEast, yNorth, zUp = self.EcefToEnu(x, y, z, lat0, lon0, h0)