import { NavLink, Outlet } from "react-router-dom";
import { assets } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";
import { toast } from 'react-toastify';

const AdminLayout = () => {
    const { axios, navigate, admin, setAdmin } = useAppContext();

    const sidebarLinks = [
        { name: "Dashboard", path: "/admin", icon: assets.dashboard },
        { name: "Users", path: "/admin/all-users", icon: assets.user_list },
        { name: "Documents", path: "/admin/all-documents", icon: assets.document_list },
        { name: "Contacts", path: "/admin/all-contacts", icon: assets.contact_list },
        { name: "Admins", path: "/admin/all-admins", icon: assets.admin_list },
    ];

    const logout = async () => {
        try {
            const { data } = await axios.get("/api/admin/logout");
            if (data.success) {
                toast.success(data.message);
                setAdmin(null);
                navigate("/");
            } else {
                toast.error(data.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    }

    return (
        <div className="h-screen">
            <div className="flex items-center justify-between px-4 md:px-8 border-b border-gray-300 py-3 bg-white">
                <NavLink to="/admin" onClick={() => setOpen(false)}>
                    <img className="h-9" src={assets.logo} alt="logo" />
                </NavLink>
                <div className="flex items-center gap-5 text-gray-500">
                    <p>Hi! {admin.name}</p>
                    <button onClick={logout} className='border rounded-full text-sm px-4 py-1 cursor-pointer'>Logout</button>
                </div>
            </div>
            <div className="flex">
                <div className="md:w-64 w-16 border-r text-base border-gray-300 pt-4 flex flex-col ">
                    {sidebarLinks.map((item) => (
                        <NavLink to={item.path} key={item.name} end={item.path === "/admin"}
                            className={({ isActive }) => `flex items-center py-3 px-4 gap-3 ${isActive ? "border-r-4 md:border-r-[6px] bg-primary/10 border-primary text-primary" : "hover:bg-gray-100/90 border-white"}`}
                        >
                            <img src={item.icon} alt={item.name} className="w-7 h-7" />
                            <p className="md:block hidden text-center">{item.name}</p>
                        </NavLink>
                    ))}
                </div>
                <Outlet />
            </div>
        </div>
    );
};

export default AdminLayout;
