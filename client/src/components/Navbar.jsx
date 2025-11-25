import { useEffect, useState } from "react"
import { NavLink } from "react-router-dom"
import { useAppContext } from "../context/AppContext";
import { assets } from "../assets/assets";
import ConfirmLogout from "./ConfirmLogout";

const Navbar = () => {
    const [open, setOpen] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);
    const { user, setShowUserLogin, navigate } = useAppContext();

    const getActivePage = ({ isActive }) => ({
        color: isActive ? "#C23700" : "#000",
    });

    return (
        <nav className="container mx-auto relative flex items-center justify-between px-6 md:px-16 lg:px-24 xl:px-32 py-4 border-b border-gray-300 bg-white transition-all">

            <NavLink to="/" onClick={() => setOpen(false)}>
                <img className="h-10" src={assets.logo} alt="logo" />
            </NavLink>

            <div className="hidden sm:flex items-center gap-8 text-nowrap">
                <NavLink to="/" style={getActivePage}>Home</NavLink>
                <NavLink to="/tools" style={getActivePage}>Tools</NavLink>
                <NavLink to="/about-us" style={getActivePage}>About Us</NavLink>
                <NavLink to="/case-studies" style={getActivePage}>Case Studies</NavLink>
                <NavLink to="/contact" style={getActivePage}>Contact</NavLink>
            </div>
            {
                !user ? (
                    <button onClick={() => setShowUserLogin(true)} className="cursor-pointer px-8 py-2 bg-special-red/90 hover:bg-special-red transition text-white rounded-full">
                        Login
                    </button>
                ) : (
                    <div className="relative group">
                        <img src={assets.profile_icon} alt="" width={40} />

                        <div className="hidden group-hover:block absolute top-0 right-0 py-10 z-40">
                            <ul className=" shadow bg-white border border-gray-200  w-30 rounded-md text-sm">
                                <li onClick={() => navigate("/my-reports")} className="p-1.5 pl-3 hover:bg-special-red/10 cursor-pointer">My Reports</li>
                                <li onClick={() => { setShowConfirm(true) }} className="p-1.5 pl-3 hover:bg-special-red/10 text-special-red cursor-pointer">Logout</li>
                            </ul>
                        </div>
                        {
                            showConfirm && <ConfirmLogout item={{ setShowConfirm }} />
                        }
                    </div>
                )
            }

            <div className="flex items-center gap-6 sm:hidden">
                <button onClick={() => setOpen(!open)} aria-label="Menu" className=" cursor-pointer">
                    <img src={assets.menu_icon} alt="menu" />
                </button>
            </div>

            {/* Mobile Menu */}
            {open && (
                <div className={`${open ? 'flex' : 'hidden'} fixed top-18 left-0 w-full h-screen bg-white/60 backdrop-blur-xl py-6 flex-col items-center gap-6 px-5 md:hidden z-50 border-t border-gray-300`}>
                    <NavLink to="/" onClick={() => setOpen(false)} style={getActivePage} >Home</NavLink>
                    <NavLink to="/my-reports" onClick={() => setOpen(false)} style={getActivePage} >My Reports</NavLink>
                    <NavLink to="/tools" onClick={() => setOpen(false)} style={getActivePage} >Tools</NavLink>
                    <NavLink to="/about-us" onClick={() => setOpen(false)} style={getActivePage} >About Us</NavLink>
                    <NavLink to="/case-studies" onClick={() => setOpen(false)} style={getActivePage}>Case Studies</NavLink>
                    <NavLink to="/contact" onClick={() => setOpen(false)} style={getActivePage} >Contact</NavLink>

                    {
                        user && (
                            <button onClick={() => { setOpen(false); setShowConfirm(true) }} className="cursor-pointer px-6 py-2 mt-2 bg-special-red/90 hover:bg-special-red transition text-white rounded-full text-sm">
                                Logout
                            </button>
                        )
                    }
                </div>
            )}

        </nav>
    )
}

export default Navbar;
