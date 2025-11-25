import { Route, Routes, useLocation } from 'react-router-dom';
import './App.css';
import { ToastContainer } from 'react-toastify';

import { useAppContext } from './context/AppContext';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ErrorPage from './pages/ErrorPage';
import Home from './pages/Home';
import Login from './components/Login';
import AboutUs from './pages/AboutUs';
import ContactPage from './pages/ContactPage';
import ViewDocument from './pages/ViewDocument';
import ToolsLayout from './pages/Tools/ToolsLayout';
import AdminLogin from './components/Admin/AdminLogin';
import AdminDocuments from './pages/Admin/AdminDocuments';
import AdminContacts from './pages/Admin/AdminContacts';
import AdminList from './pages/Admin/AdminList';
import AdminUsers from './pages/Admin/AdminUsers';
import AdminDashboard from './pages/Admin/AdminDashboard';
import AdminLayout from './pages/Admin/AdminLayout';
import CaseStudiesPage from './pages/CaseStudiesPage';
import MyReports from './pages/MyReports';
import UploadMRIReport from './pages/Tools/UploadMRIReport';
import { useState } from 'react';
import { MessageCircle } from 'lucide-react';
import ChatBot from './components/ChatBot';

const App = () => {
  const isAdminPath = useLocation().pathname.includes("admin");
  const [open, setOpen] = useState(false);
  const { showUserLogin, showAdminLogin } = useAppContext();

  return (
    <div className='text-default min-h-screen text-gray-700 bg-white'>
      {isAdminPath ? null : <Navbar />}
      {showUserLogin ? <Login /> : null}

      <ToastContainer />

      <div className={`${isAdminPath ? "" : ""}`}>
        <Routes>
          <Route path='*' element={<ErrorPage />} />
          <Route path='/' element={<Home />} />
          <Route path='/my-reports' element={<MyReports />} />
          <Route path='/my-reports/:id' element={<ViewDocument />} />
          <Route path='/tools' element={<ToolsLayout />} />
          <Route path='/tools' >
            <Route path='brain-tumor-detection' element={<UploadMRIReport />} />
            <Route path='image-preprocessing' element={<div className='h-screen text-center text-xl mt-60 md:mt-80'>Working on Tool...</div>} />
            <Route path='report-generation' element={<div className='h-screen text-center text-xl mt-60 md:mt-80'>Working on Tool...</div>} />
          </Route>
          <Route path='/about-us' element={<AboutUs />} />
          <Route path='/case-studies' element={<CaseStudiesPage />} />
          <Route path='/contact' element={<ContactPage />} />

          {/* Admin */}
          <Route path='/admin' element={showAdminLogin ? <AdminLogin /> : <AdminLayout />}>
            <Route index element={showAdminLogin ? null : <AdminDashboard />} />
            <Route path='all-users' element={showAdminLogin ? null : <AdminUsers />} />
            <Route path='all-documents' element={showAdminLogin ? null : <AdminDocuments />} />
            <Route path='all-contacts' element={showAdminLogin ? null : <AdminContacts />} />
            <Route path='all-admins' element={showAdminLogin ? null : <AdminList />} />
          </Route>

        </Routes>
      </div>

      <div>
        <button
          onClick={() => setOpen(!open)}
          className="fixed bottom-5 right-5 bg-special-red/90 cursor-pointer hover:bg-special-red text-white p-4 rounded-full shadow-lg transition-transform transform hover:scale-110"
        >
          <MessageCircle className="w-6 h-6" />
        </button>

        {
          open && (
            <div className={`fixed bottom-20 right-5 md:w-96 rounded-2xl overflow-hidden border border-special-red/40 z-50 ${open ? "animate-fade-in-up transition-all duration-300" : "animate-fade-out-down transition-all duration-300"}`}>
              <ChatBot />
            </div>
          )
        }
      </div>

      {isAdminPath ? null : <Footer />}
    </div>
  )
}

export default App;
