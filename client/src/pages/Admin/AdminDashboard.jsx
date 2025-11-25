import React, { useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import { MaterialReactTable } from 'material-react-table';

const AdminDashboard = () => {
  const {
    documentList, getAllDocumentsByAdmin,
    userList, getAllUsersByAdmin,
    adminList, getAllAdminsByAdmin,
    contactList, getAllContactsByAdmin
  } = useAppContext();

  useEffect(() => {
    getAllDocumentsByAdmin();
    getAllUsersByAdmin();
    getAllAdminsByAdmin();
    getAllContactsByAdmin();
  }, []);

  const documentColumns = [
    { accessorKey: "_id", header: "Document ID" },
    {
      accessorKey: "image",
      header: "Image",
      Cell: ({ cell }) => <img src={cell.getValue()} alt="doc" width="50" height="50" />,
    },
    { accessorKey: "name", header: "Name" },
    { accessorKey: "userId", header: "User ID" },
  ];

  const userColumns = [
    { accessorKey: "_id", header: "User ID" },
    { accessorKey: "name", header: "Name" },
    { accessorKey: "email", header: "Email" }
  ];

  const adminColumns = [
    { accessorKey: "_id", header: "Admin ID" },
    { accessorKey: "name", header: "Name" },
    { accessorKey: "email", header: "Email" },
  ];

  const contactColumns = [
    { accessorKey: "_id", header: "Contact ID" },
    { accessorKey: "name", header: "Name" },
    { accessorKey: "email", header: "Email" },
    { accessorKey: "message", header: "Message" },
  ];

  return (
    <div className="overflow-y-scroll w-full h-screen p-6 bg-gray-100">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Documents */}
        <Section title="Recent Documents" columns={documentColumns} data={documentList} />
        
        {/* Users */}
        <Section title="Registered Users" columns={userColumns} data={userList} />

        {/* Admins */}
        <Section title="Admin Accounts" columns={adminColumns} data={adminList} />

        {/* Contacts */}
        <Section title="Contact Queries" columns={contactColumns} data={contactList} />
      </div>
    </div>
  );
};

const Section = ({ title, columns, data }) => (
  <div className="bg-white p-4 rounded-lg shadow-md h-[70vh] lg:h-[45vh] flex flex-col">
    <div className="flex justify-between items-center mb-2">
      <h2 className="text-lg font-semibold">{title}</h2>
    </div>
    <div className="overflow-auto flex-1">
      <MaterialReactTable columns={columns} data={data} />
    </div>
  </div>
);

export default AdminDashboard;

