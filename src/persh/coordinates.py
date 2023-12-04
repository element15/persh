"""Utilities for dealing with coordinates and rotations."""

import numpy as np
try:
    from numba import njit, guvectorize
except ModuleNotFoundError:
    # Make dummy decorators so that Numba is effectively bypassed since it
    # isn't installed.
    njit = lambda *args, **kwargs: lambda f: f
    # Bypassing `guvectorize` is a bit more involved, since the functions
    # don't directly return results.
    def guvectorize(*args, **kwargs):
        def devectorize(f):
            def wrapper(pos):
                pos = np.asarray(pos)
                res = np.empty(pos.shape)
                res_2d = res.view()
                res_2d.shape = (-1, 3) # Exception if not in-place
                pos_2d = pos.reshape((-1, 3))
                for pos_row, res_row in zip(pos_2d, res_2d):
                    f(pos_row, res_row)
                return res
            return wrapper
        return devectorize

import math

## Constants ##

# Skew-symmetric matrices
@njit(cache=True)
def skew(x, y, z):
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]])
def deskew(O):
    return np.array([O[2, 1], O[0, 2], O[1, 0]])

# Earth rate
OMEGA_E = 72.92115e-6 # Rotation rate of the earth (rad/s)
OMEGA_E_VEC = np.array([0.0, 0.0, OMEGA_E])
OMEGA_E_SQ = OMEGA_E * OMEGA_E
OMEGA_E_SKEW = skew(*OMEGA_E_VEC)

# WGS84 ellipsoid constants
WGS84_R_EQ = 6_378_137.0 # WGS84 equitorial radius
WGS84_E = 0.081_819_190_842_5  # WGS84 eccentricity
WGS84_E_SQ = WGS84_E * WGS84_E
WGS84_A = 6_378_137.0 # WGS84 semi-major axis
WGS84_A_SQ = WGS84_A * WGS84_A
WGS84_B = 6_356_752.31424518 # WGS84 semi-minor axis
WGS84_B_SQ = WGS84_B * WGS84_B
WGS84_E_PRIME_SQ = (WGS84_A_SQ - WGS84_B_SQ) / WGS84_B_SQ
WGS84_E_PRIME_SQ_B = WGS84_E_PRIME_SQ * WGS84_B
WGS84_E_SQ_A = WGS84_E_SQ * WGS84_A
WGS84_OM_E_SQ = 1.0 - WGS84_E_SQ
WGS84_OM_E_SQ_SQ = WGS84_OM_E_SQ * WGS84_OM_E_SQ

# Multiply a quaternion by this to get the conjugate
CONJ_MULTIPLIER = np.array([1.0, -1.0, -1.0, -1.0])

# IMU unit conversions
DEG_HR_TO_RAD_S = math.pi / (180.0 * 3600.0) # Used for gyro bias/drift
DEG_RT_HR_TO_RAD_S_RT_HZ = math.pi / (180.0 * 60) # Used for ARW
MG_TO_M_S2 = 9.81 / 1e3
FPS_RT_HR_TO_MPS2_RT_HZ = 1.0 / (0.8128 * 60.0) # Used for VRW

hav = lambda x: 0.5*(1.0 - np.cos(x))
archav = lambda x: np.arccos(1.0 - 2.0*x)

# Degree trig functions
_to_rad = lambda f: lambda x: f(np.radians(x))
_to_deg = lambda f: lambda x: np.degrees(f(x))
RAD_TO_DEG = 180.0 / math.pi
RAD_TO_DEG_LLA_MULTIPLIER = np.array([RAD_TO_DEG, RAD_TO_DEG, 1.0])

# Wrap a function in np.asarray
_asarray = lambda f: lambda x: f(np.asarray(x))

sind, cosd, tand, havd = map(_to_rad, (np.sin, np.cos, np.tan, hav))
arcsind, arccosd, archavd = map(_to_deg, (np.arcsin, np.arccos, archav))

# Semicircle trig functions
cos_semi = lambda x: np.cos(x * np.pi)
sin_semi = lambda x: np.sin(x * np.pi)

# Elementary rotation matrices
@njit(cache=True)
def rot_z(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return np.array([
        [c_theta, -s_theta, 0.0],
        [s_theta, c_theta, 0.0],
        [0.0, 0.0, 1.0]])
@njit(cache=True)
def rot_y(theta):
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)
    return np.array([
        [c_theta, 0.0, s_theta],
        [0.0, 1.0, 0.0],
        [-s_theta, 0.0, c_theta]])
@njit(cache=True)
def rot_x(theta):
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c_theta, -s_theta],
        [0.0, s_theta, c_theta]])
@njit(cache=True)
def rot_2(theta):
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)
    return np.array([
        [c_theta, -s_theta],
        [s_theta, c_theta]])
rot_xd, rot_yd, rot_zd, rot_2d = map(_to_rad, (rot_x, rot_y, rot_z, rot_2))

@njit(cache=True)
def rotmat_euler(C):
    """Convert a rotation matrix to ZYX Euler angles.

    Parameters:
    C - Rotation matrix (direction cosine matrix or DCM)

    Returns:
    Euler angles as a Numpy array with shape (3,). The *rotation* order of the
    angles is ZYX (also, 321), but the angles are *sotred* in XYZ (roll, pitch,
    yaw) order.

    Note that each row of `C`  will be divided through by the 2-norm of that
    row before conversion occurs. This step is intended to combat issues
    arising from small numeric issues which may arise from combining rotations
    together. Beyond this rudimentary step, it is assumed that `C` is a
    well-formed and valid rotation matrix.
    """
    # Normalize the rotation matrix first:
    # C = C / np.linalg.norm(C, axis=0) # Does not work with Numba
    C = C / np.sqrt(np.sum(C * C, axis=0))
    # Groves Eq. 2.17
    phi = np.arctan2(C[2, 1], C[2, 2])
    theta = -np.arcsin(C[2, 0])

    psi = np.arctan2(C[1, 0], C[0, 0]) # Original
    # psi = np.arctan2(Cbe[0, 0], Cbe[1, 0]) # From Dr. Martin

    eulers = np.array([phi, theta, psi])

    return eulers

@njit(cache=True)
def euler_rotmat(eulers):
    """Convert a set of ZYX Euler angles to a rotation matrix.

    Parameters:
    eulers - Euler angles with rotation order ZYX (or 321), stored in XYZ
        (roll, pitch, yaw) order

    Returns:
    Rotation matrix (direction cosine matrix, DCM) as a NumPy array with shape
    (3, 3).
    """
    c_phi, c_theta, c_psi = map(math.cos, eulers)
    s_phi, s_theta, s_psi = map(math.sin, eulers)

    # Groves Eq. 2.14
    r00 = c_theta*c_psi
    r01 = -c_phi*s_psi + s_phi*s_theta*c_psi
    r02 = s_phi*s_psi + c_phi*s_theta*c_psi

    r10 = c_theta*s_psi
    r11 = c_phi*c_psi + s_phi*s_theta*s_psi
    r12 = -s_phi*c_psi + c_phi*s_theta*s_psi

    r20 = -s_theta
    r21 = s_phi * c_theta
    r22 = c_phi * c_theta

    C = np.array((
        (r00, r01, r02),
        (r10, r11, r12),
        (r20, r21, r22)))

    return C

@njit(cache=True)
def ned_ecef_rotmat(lat, lon):
    """Produce the rotation matrix which rotates NED to ECEF.

    Parameters:
    lat - Reference latitude of the NED frame
    lon - Reference longitude of the NED frame
    """
    # Groves Eq. 2.158 (2nd ed)
    c_lat = math.cos(lat)
    c_lon = math.cos(lon)
    s_lat = math.sin(lat)
    s_lon = math.sin(lon)

    Cle = np.array(( # Rotation matrix: Local -> ECEF
        (-s_lat*c_lon, -s_lon, -c_lat*c_lon),
        (-s_lat*s_lon,  c_lon, -c_lat*s_lon),
        ( c_lat,          0.0, -s_lat      )))
    return Cle

ned_ecef_rotmat_deg = lambda lat, lon: ned_ecef_rotmat(*np.radians((lat, lon)))

@njit(cache=True)
def ned_enu(ned):
    """Convert NED to/from ENU.

    Note that since the processes for converting NED to ENU and ENU to NED are
    identical, this function will work equally well in either direction.
    """
    ned = np.asarray(ned)
    n, e, d = ned.T
    enu = np.array((e, n, -d)).T
    return enu

enu_ned = ned_enu

def lla_radians(lla_degrees):
    """Convert only the angluar parts of an LLA location to radians."""
    return lla_degrees / RAD_TO_DEG_LLA_MULTIPLIER

def lla_degrees(lla_radians):
    """Convert only the angular parts of an LLA location to degrees."""
    return lla_radians * RAD_TO_DEG_LLA_MULTIPLIER

def ecef_ned(r_eb_e, lla_ref=None, xyz_ref=None, degrees=False):
    """Convert an ECEF location to NED.

    Parameters:
    r_eb_e - ECEF to body vector resolved in ECEF
        (This can be a single location or an nx3 array of locations.)
    lla_ref - LLA location of the NED frame
    xyz_ref - ECEF location of the NED frame
    degrees - Indicates whether the LLA reference is in radians or degrees

    Note: At least one of `lla_ref` or `xyz_ref` must be specified. If only one
    is specified, the other will be calculated. This is recommended in most
    cases as it is less error prone. That said, if the NED frame location is
    already calculated in both LLA and ECEF, both can be passed in and the
    conversion step will be skipped. Do this only if performance is a major
    concern and many reference frame conversions are being done.
    """
    r_eb_e = np.asarray(r_eb_e)
    if lla_ref is None and xyz_ref is None:
        raise TypeError(
            'At least one of `lla_ref` or `xyz_ref` must be given.')
    if lla_ref is not None and degrees:
        lla_ref = lla_ref / RAD_TO_DEG_LLA_MULTIPLIER
    r_el_e = (
        lla_ecef(lla_ref) if xyz_ref is None else xyz_ref)
    lat, lon, _ = (
        ecef_lla(xyz_ref) if lla_ref is None else lla_ref)
    r_lb_e = r_eb_e - r_el_e
    Cel = ned_ecef_rotmat(lat, lon).T
    r_lb_l = (Cel@np.expand_dims(r_lb_e, -1)).squeeze(-1)
    return r_lb_l


def ned_ecef(r_lb_l, lla_ref=None, xyz_ref=None, degrees=False):
    """Convert an NED location to ECEF.

    Paremeters:
    r_lb_l - NED to body vector resolved in NED
        (This can be a single location or an nx3 array of locations.)
    lla_ref - LLA location of the NED frame
    xyz_ref - ECEF location of the NED frame
    degrees - Indicates whether the LLA reference is in radians or degrees

    See the documentation for `ecef_ned()` for an important note on the
    behavior of `lla_ref` and `xyz_ref`.
    """
    r_lb_l = np.asarray(r_lb_l)
    if lla_ref is None and xyz_ref is None:
        raise TypeError(
            'At least one of `lla_ref` or `xyz_ref` must be given.')
    if lla_ref is not None and degrees:
        lla_ref = lla_ref / RAD_TO_DEG_LLA_MULTIPLIER
    r_el_e = (
        lla_ecef(lla_ref) if xyz_ref is None else xyz_ref)
    lat, lon, _ = (
        ecef_lla(xyz_ref) if lla_ref is None else lla_ref)
    Cle = ned_ecef_rotmat(lat, lon)
    r_lb_e = (Cle@np.expand_dims(r_lb_l, -1)).squeeze(-1)
    r_eb_e = r_el_e + r_lb_e
    return r_eb_e

@guvectorize(['void(float64[:], float64[:])'], '(n)->(n)', nopython=True)
def ecef_lla(pos, res):
    """Convert an ECEF location to a WGS84 LLA location.

    Parameters:
    pos - ECEF XYZ coordinate (m) (Broadcastable over an array of positions)
    """
    assert pos.shape[-1] == 3
    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]
    p = np.hypot(pos_x, pos_y)
    theta = np.arctan2(pos_z * WGS84_A, p * WGS84_B) # intermediate variable
    lat_ref = np.arctan2(
        pos_z + WGS84_E_PRIME_SQ_B * np.sin(theta)**3,
        p - WGS84_E_SQ_A * np.cos(theta)**3) # convert ECEF to Latitude
    lon_ref = np.arctan2(pos_y, pos_x)

    # sin(lat_ref)**2, cos(lat_ref)**2
    sin_lat_ref_2 = np.sin(lat_ref)
    sin_lat_ref_2 *= sin_lat_ref_2
    cos_lat_ref = np.cos(lat_ref)

    transverse_scale = np.sqrt(1.0 - WGS84_E_SQ * sin_lat_ref_2)

    if p < 1e-3: # Small angle approximation for locations near poles
        if pos_z < 1e-3: # Within 1 mm of the origin, just return origin
            res[:] = np.array([0.0, 0.0, -WGS84_A])
            return
        alt_m = pos_z - WGS84_B
    else:
        alt_m = p / cos_lat_ref - WGS84_A / transverse_scale

    res[0] = lat_ref
    res[1] = lon_ref
    res[2] = alt_m

ecef_lla_deg = lambda pos: ecef_lla(pos) * RAD_TO_DEG_LLA_MULTIPLIER

@guvectorize(['void(float64[:], float64[:])'], '(n)->(n)', nopython=True)
def lla_ecef(pos, res):
    """Convert a WGS84 LLA location to ECEF (meters).

    Parameters:
    pos - LLA coordinate (rad, rad, m)
        (Broadcastable over an array of positions)
    """
    assert pos.shape[-1] == 3
    lat = pos[0]
    lon = pos[1]
    alt = pos[2]
    c_lat = np.cos(lat)
    s_lat = np.sin(lat)
    c_lon = np.cos(lon)
    s_lon = np.sin(lon)

    re = WGS84_R_EQ / np.sqrt(1.0 - WGS84_E_SQ * s_lat*s_lat) # Groves 2.106
    re_h = re + alt

    x = re_h * c_lat * c_lon
    y = re_h * c_lat * s_lon
    z = (WGS84_OM_E_SQ * re + alt) * s_lat

    res[0] = x
    res[1] = y
    res[2] = z

lla_ecef_deg = lambda pos: lla_ecef(pos / RAD_TO_DEG_LLA_MULTIPLIER)

def ned_lla(r_lb_l, lla_ref=None, xyz_ref=None, degrees=False):
    """Convert an NED location to WGS84 LLA.

    This function passes all arguments directly to `ned_ecef`, then passes
    the resulting ECEF position to `ecef_lla`. Refer to the documentation for
    those functions for more information.
    """
    r_eb_e = ned_ecef(r_lb_l, lla_ref, xyz_ref, degrees)
    ecef_lla_func = ecef_lla_deg if degrees else ecef_lla
    r_lla = ecef_lla_func(r_eb_e)
    return r_lla

def lla_ned(pos_lla, lla_ref=None, xyz_ref=None, degrees=False):
    """Convert a WGS84 LLA position to NED.

    This function delegates to `lla_ecef`, then passes all arguments to
    `ecef_ned`. Refer to the documentation for those functions for more
    information.
    """
    lla_ecef_func = lla_ecef_deg if degrees else lla_ecef
    r_eb_e = lla_ecef_func(pos_lla)
    r_lb_l = ecef_ned(r_eb_e, lla_ref, xyz_ref, degrees)
    return r_lb_l

## JIT-compiled quaternion functions (Called from the `quaternion` class) ##
@njit(cache=True)
def _q_from_rotmat(C):
    """See `quaternion.from_rotmat()`."""
    q_w = 0.5 * np.sqrt(1 + np.trace(C))
    if q_w > 0.01:
        q_w_4 = 4 * q_w
        q_x = (C[2, 1] - C[1, 2]) / q_w_4
        q_y = (C[0, 2] - C[2, 0]) / q_w_4
        q_z = (C[1, 0] - C[0, 1]) / q_w_4
    else:
        q_x = 0.5 * np.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2])
        q_x_4 = 4 * q_x
        q_w = (C[2, 1] - C[1, 2]) / q_x_4
        q_y = (C[0, 1] + C[1, 0]) / q_x_4
        q_z = (C[0, 2] + C[2, 0]) / q_x_4

    q = np.array([q_w, q_x, q_y, q_z])
    return q

@njit(cache=True)
def _q_from_angle_axis(angle, axis):
    """See `quaternion.from_angle_axis()`."""
    half_ang = angle / 2.0
    u = np.asarray(axis) / np.linalg.norm(axis)
    q_v = np.sin(half_ang) * u
    q_w = np.cos(half_ang)
    q = np.array([q_w, q_v[0], q_v[1], q_v[2]])
    return q

@njit(cache=True)
def _q_from_euler(eulers):
    """See `quaternion.from_euler()`."""
    half_eulers = eulers / 2.0
    c_phi, c_theta, c_psi = np.cos(half_eulers)
    s_phi, s_theta, s_psi = np.sin(half_eulers)

    q_w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    q_x = - s_phi * c_theta * c_psi + c_phi * s_theta * s_psi
    q_y = - c_phi * s_theta * c_psi - s_phi * c_theta * s_psi
    q_z = - c_phi * c_theta * s_psi + s_phi * s_theta * c_psi

    q = np.array([q_w, q_x, q_y, q_z])
    return q

@njit(cache=True)
def _q_to_rotmat(q):
    """See `quaternion.to_rotmat()`."""
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]

    q_ww = q_w * q_w
    q_wx = q_w * q_x
    q_wy = q_w * q_y
    q_wz = q_w * q_z
    q_xx = q_x * q_x
    q_xy = q_x * q_y
    q_xz = q_x * q_z
    q_yy = q_y * q_y
    q_yz = q_y * q_z
    q_zz = q_z * q_z

    r00 = q_ww + q_xx - q_yy - q_zz
    r01 = 2 * (q_xy - q_wz)
    r02 = 2 * (q_xz + q_wy)

    r10 = 2 * (q_xy + q_wz)
    r11 = q_ww - q_xx + q_yy - q_zz
    r12 = 2 * (q_yz - q_wx)

    r20 = 2 * (q_xz - q_wy)
    r21 = 2 * (q_yz + q_wx)
    r22 = q_ww - q_xx - q_yy + q_zz

    C = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]])
    return C

@njit(cache=True)
def _q_to_euler(q):
    """See `quaternion.to_euler()`."""
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]

    q_yy2 = 2.0 * q_y*q_y # This is the only term to appear twice
    phi = np.arctan2(
        2.0 * (q_y*q_z - q_w*q_x), 1.0 - 2.0 * q_x*q_x - q_yy2)
    theta = np.arcsin(2.0 * (-q_w*q_y - q_x*q_z))
    psi = np.arctan2(
        2.0 * (q_x*q_y - q_w*q_z), 1 - q_yy2 - 2.0 * q_z*q_z)

    eulers = np.array([phi, theta, psi])
    return eulers

@njit(cache=True)
def _q_left_product_matrix(q):
    """See `quaternion.left_product_matrix()`."""
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]
    q_l = np.array([
        [q_w, -q_x, -q_y, -q_z],
        [q_x, q_w, -q_z, q_y],
        [q_y, q_z, q_w, -q_x],
        [q_z, -q_y, q_x, q_w]])
    return q_l

class quaternion(np.ndarray):
    """A unit quaternion for representing an attitude.

    These quaternions are assumed to follow Hamiltonian conventions. Namely:
        - Element order is [real, i, j, k]
        - Quaternion is right handed (i*j = k, i*j*k = -1)

    Resources and references (with associated caveats):
    1. "Quaternion kinematics for the error-state Kalman filter", by Joan Solà
        (arXiv article)

        This source does an excellent job at delineating *exactly* what the
        different quaternion conventions are and which ones are being used. The
        convention implemented here *should* match Solà's (i.e. the Hamiltonian
        convention).

    2. "Exponential Map, Angle Axis, and Angular Velocity", by Daniel Holden
        (Personal blog, `theorangeduck.com`)

        This relatively brief article acts as a companion to the Solà text,
        offering some more concrete examples and a more approachable
        explanation of the quaternion exponential, which is important when
        integrating angular velocities.

    3. "Principles of GNSS, Inertial, and Multisensor Integrated Navigation
        Systems", by Paul D. Groves (Textbook)

        Groves doesn't make much use of quaternions in his work, but the text
        does provide a handful of conversion formulas to/from other attitude
        representations. The text was used as a cross-reference for other
        sources, and as the main source for the conversion formulas to/from
        Euler angles. The other listed sources do not contain much information
        on Euler angles. Do keep in mind that Groves's notation can be quite
        dense, and that his notation for the direction of a quaternion's
        transformation is backwards compared to his notation for the equivalent
        rotation matrix.
    """

    def __new__(cls, value):
        obj = np.asarray(value).view(cls)
        return obj

    def __getitem__(self, items):
        """Override slicing behavior so that a normal ndarray is returned.

        Extracting a subset of the elements of a quaternion no longer
        constitutes a quaternion. As such, having an array usually makes more
        sense.
        """
        return super().__getitem__(items).view(np.ndarray)

    @classmethod
    def from_rotmat(cls, C):
        """Create a quaternion representing the same rotation as `C`.

        Suppose rotation `C` represents a rotation from frame `alpha` to frame
        `beta` when multiplied from the left. In other words:
            `x_beta = C @ x_alpha`.
        The resulting quaternion, `q`, will produce the same rotation when
        multiplied normally from the left, and conjugated from the right:
            `x_beta = q (x) x_alpha (x) q_conj`.
        """
        return _q_from_rotmat(np.asarray(C)).view(cls)

    @classmethod
    def from_angle_axis(cls, angle, axis, degrees=False):
        """Create a quaternion corresponding to an angle/axis rotation.

        The axis need not necessarily be a unit vector.
        """
        if degrees:
            angle = np.radians(angle)

        return _q_from_angle_axis(angle, np.asarray(axis)).view(cls)

    @classmethod
    def from_euler(cls, eulers, degrees=False):
        """Get a quaternion from the equivalent ZYX Euler angles."""
        if degrees:
            eulers = np.radians(eulers)

        return _q_from_euler(np.asarray(eulers)).view(cls)

    @classmethod
    def exp_small_ang(cls, v):
        """Direct access to the small angle quaternion exponential.

        This may be useful in performance-critical applications such as angular
        velocity integration.
        """
        q_base = np.array([1.0, *v])
        return (q_base / np.linalg.norm(q_base)).view(cls)

    @classmethod
    def exp(cls, v):
        """Quaternion exponential."""
        half_angle = np.linalg.norm(v)
        if half_angle < 1e-6: # Avoid numerical issues with small angle approx
            return cls.exp_small_ang(v)
        q_w = np.cos(half_angle)
        q_v = np.sin(half_angle) / half_angle * v
        return cls([q_w, *q_v])

    def to_rotmat(self):
        return _q_to_rotmat(self.view(np.ndarray))

    def angle(self, degrees=False):
        """Get only the angle of rotation."""
        angle = 2.0 * np.arctan2(np.linalg.norm(self.imag), self.real)

        if degrees:
            angle = np.degrees(angle)

        return angle

    def to_angle_axis(self, degrees=False):
        q_v = self.imag
        l = np.linalg.norm(q_v)
        if l < 1e-6: # Assume null rotation to avoid numerical issues
            # TODO: Replace this with a truncated Taylor series similar to that
            # used in `quaternion.log()`.
            return 0.0, np.array([1.0, 0.0, 0.0])
        angle = 2.0 * np.arctan2(l, self.real)
        axis = q_v / l

        if degrees:
            angle = np.degrees(angle)

        return angle, axis

    def log(self):
        """Quaternion logarithm.

        The quaternion logarithm in this instance corresponds to the axis of
        rotation scaled by the half angle. This operation is, as might be
        expected, the inverse of the quaternion exponential.
        """
        q_w = self.real
        q_v = self.imag
        l = np.linalg.norm(q_v)
        half_angle = np.arctan2(l, q_w)
        if half_angle < 1e-6:
            v = q_v/q_w * (1.0 - l*l / (3.0 * q_w*q_w))
            return v
        u = q_v / l
        return half_angle * u

    def to_euler(self, degrees=False):
        """Get the ZYX Euler angle representation."""
        eulers = _q_to_euler(self.view(np.ndarray))

        if degrees:
            eulers = np.degrees(eulers)

        return eulers

    @property
    def left_product_matrix(self):
        """Produce the matrix corresponding to a left multiplication.

        This is a 4x4 matrix which can be matrix multiplied by another
        quaternion to produce matrix multiplication.
        """
        return _q_left_product_matrix(self.view(np.ndarray))

    @property
    def imag(self):
        """Get the complex part of the quaternion as a length-3 array."""
        return self[1:]

    @property
    def real(self):
        """Get the real part of the quaternion."""
        return self[0]

    @property
    def conj(self):
        """Get the conjugate of the quaternion."""
        c = self * CONJ_MULTIPLIER
        return c

    def left_product(self, q2):
        q_prod = self.left_product_matrix @ q2
        return q_prod.view(quaternion)

if __name__ == '__main__':
    # Run test suite
    print('Running self-test...')

    ## Coordinate transform tests ##
    # These tests mostly amount to taking some test locations and converting
    # them through several different coordinate representations, ensuring that
    # they will successfully 'round trip' and convert back to their original
    # value.
    print('Testing ECEF/LLA conversions...')
    test_locations_lla = np.array([ # deg, deg, m
        [0, 0, 0],
        [32.6, -85.5, 200],
        [0, 120, 0],
        [35, -102, -80],
        [89, 0, 0],
        [89, 75, 0],
        [90, 0, 0],
        [90, 60, 0],
        [-60, 30, 0]])
    test_locations_ecef = np.array([
        [0, 0, 0],
        [-1.8805e6, 1.5114e6, 5.8847e6],
        [1e6, 0, 0],
        [0, 1e6, 0],
        [0, 0, 1e6]])

    for test_location_lla in test_locations_lla:
        print(f'Round-trip testing LLA: {test_location_lla}...', end='')
        round_trip = ecef_lla_deg(lla_ecef_deg(test_location_lla))
        if np.allclose(round_trip, test_location_lla, atol=1e-6):
            print('PASS')
        else:
            print(
                f'FAIL\n'
                f'    Input LLA: {test_location_lla}\n'
                f'    Output LLA: {round_trip}')
    for test_location_ecef in test_locations_ecef:
        print(f'Round-trip testing ECEF: {test_location_ecef}...', end='')
        round_trip = lla_ecef_deg(ecef_lla_deg(test_location_ecef))
        if np.allclose(
            round_trip, test_location_ecef,
            rtol=0, atol=1e-6):
            print('PASS')
        else:
            print(
                f'FAIL\n'
                f'    Input ECEF: {test_location_ecef}\n'
                f'    Output ECEF: {round_trip}')
