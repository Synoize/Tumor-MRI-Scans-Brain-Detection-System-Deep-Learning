import { createContext, useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import axios from "axios";

axios.defaults.withCredentials = true;
axios.defaults.baseURL = import.meta.env.VITE_BACKEND_URL;

export const AppContext = createContext();

export const AppContextProvider = ({ children }) => {

    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [user, setUser] = useState(null);
    const [admin, setAdmin] = useState(null);
    const [showAdminLogin, setShowAdminLogin] = useState(true);
    const [showUserLogin, setShowUserLogin] = useState(false);
    const [myDocuments, setMyDocuments] = useState([]);
    const [documentList, setDocumentList] = useState([]);
    const [userList, setUserList] = useState([]);
    const [adminList, setAdminList] = useState([]);
    const [contactList, setContactList] = useState([]);
    const [banUser, setBanUser] = useState({});

    // Fatch Admin Status
    const authUser = async () => {
        try {
            const { data } = await axios.get('/api/user/auth');
            if (data.success) {
                setUser(data.user);
            }
        } catch (error) {
            setUser(null);
        }
    }

    // Fatch Admin Status
    const authAdmin = async () => {
        try {
            const { data } = await axios.get('/api/admin/auth');
            if (data.success) {
                setAdmin(data.admin);
                setBanUser(data.admin.banUser);
                setShowAdminLogin(false);
            }
        } catch (error) {
            setAdmin(null);
        }
    }

    //Get My All Documents

    const getMyDocuments = async () => {
        try {
            const { data } = await axios.get('/api/document/user/docs', {
                params: { userId: user._id }
            });

            if (data?.success) {
                setMyDocuments(data?.document);

            } else {
                toast.error(data?.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    };


    // Get All Docments of User by Admin
    const getAllDocumentsByAdmin = async () => {
        try {
            const { data } = await axios.get('/api/document/list');
            if (data?.success) {
                setDocumentList(data?.document);

            } else {
                toast.error(data?.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    };

    // Get All Users by Admin
    const getAllUsersByAdmin = async () => {
        try {
            const { data } = await axios.get('/api/admin/get-users');

            if (data?.success) {
                setUserList(data?.users);

            } else {
                toast.error(data?.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    };

    // Get All Admins by Admin
    const getAllAdminsByAdmin = async () => {
        try {
            const { data } = await axios.get('/api/admin/get-admins');

            if (data?.success) {
                setAdminList(data?.admins);

            } else {
                toast.error(data?.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    };

    // Get All Contacts by Admin
    const getAllContactsByAdmin = async () => {
        try {
            const { data } = await axios.get('/api/admin/get-contacts');

            if (data?.success) {
                setContactList(data?.contacts);

            } else {
                toast.error(data?.message);
            }
        } catch (error) {
            toast.error(error.message);
        }
    };

    useEffect(() => {
        authUser();
        authAdmin();
    }, [])

    const value = {
        toast, navigate, axios, loading, setLoading,
        user, setUser, admin, setAdmin, authUser, authAdmin,
        showAdminLogin, setShowAdminLogin, showUserLogin, setShowUserLogin,
        myDocuments, setMyDocuments, getMyDocuments,
        documentList, setDocumentList, getAllDocumentsByAdmin,
        userList, setUserList, getAllUsersByAdmin,
        adminList, setAdminList, getAllAdminsByAdmin,
        contactList, setContactList, getAllContactsByAdmin
    }
    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    );
}

export const useAppContext = () => {
    return useContext(AppContext);
}
