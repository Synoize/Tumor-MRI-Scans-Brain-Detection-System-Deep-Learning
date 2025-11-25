import React, { useEffect } from 'react'
import { useAppContext } from '../../context/AppContext';
import { MaterialReactTable } from "material-react-table";

const columns = [
  { accessorKey: "_id", header: "Contact ID" },
  { accessorKey: "name", header: "Full Name" },
  { accessorKey: "subject", header: "Subject" },
  { accessorKey: "message", header: "Message" },
  { accessorKey: "phone", header: "Phone Number" },
  { accessorKey: "email", header: "Email Id" },
  { accessorKey: "createdAt", header: "Created" },
  { accessorKey: "updatedAt", header: "Updated" },
];


const AdminContacts = () => {
  const { contactList, setContactList, getAllContactsByAdmin } = useAppContext();

  useEffect(() => {
    getAllContactsByAdmin();
  }, []);

  return (
    <div className="overflow-y-scroll h-screen w-full p-6 bg-gray-100">
      <div className="flex items-center justify-between bg-white p-3 rounded-lg shadow mb-4">
        <h2 className="text-lg font-bold text-gray-700">Contact List</h2>
      </div>
      <MaterialReactTable columns={columns} data={contactList} />
    </div>
  )
}

export default AdminContacts
